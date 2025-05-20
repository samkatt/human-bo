"""Various models and posteriors."""

from math import prod
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from human_bo import moo, utils


def create_gp(data: trieste.data.Dataset, search_space: trieste.space.SearchSpace):
    """Factory function for creating (Trieste) posterior models.

    This is just a basic GP.

    Note: will call `optimize` on the model before returning.
    """
    gp = trieste.models.gpflow.models.GaussianProcessRegression(
        trieste.models.gpflow.builders.build_gpr(
            data, search_space, trainable_likelihood=True
        )
    )
    gp.optimize(data)

    return gp


class WeightedParticles:
    """Implements basic weighted particles / ensemble."""

    def __init__(
        self,
        particles: list[Any] | tf.Tensor,
        logits: list[float] | list[tf.Tensor] | tf.Tensor,
    ):
        """Creates a distribution over `particles` according to their `logits`"""
        assert len(particles) == len(logits)
        self.particles = tf.convert_to_tensor(particles, tf.float64)
        self.normalized_weights = tf.reduce_logsumexp(logits)
        self.distr = tfp.distributions.Categorical(logits=logits)

    def sample(self, shape: list[int]) -> tf.Tensor:
        """Samples `n` `particles` according to their `logits`."""
        return tf.gather(self.particles, self.distr.sample(shape))

    def map(self) -> tf.Tensor:
        """Returns the most likely element in `particles` according to `logits`.

        Does *not care* for repeating particles (as in, will not add their weight).
        """
        return tf.gather(self.particles, self.distr.mode())


class LinearPosterior(trieste.models.interfaces.SupportsGetObservationNoise):
    """A Monte-Carlo approximation of a Trieste `ProbabilisticModel` of linear weights"""

    def __init__(
        self,
        X: tf.Tensor,
        Y: tf.Tensor,
        observation_noise: float = 0.1,
        n_approx: int = 100,
    ):
        """Creates `WeightedParticles` model, assuming linear mapping of `data`.

        :observation_noise: the assume noise of the linear function.
        :n_approx: the number of samples to approximate the distribution with.
        """
        if tf.size(X) == 0:
            raise ValueError("Cannot initiate `LinearPosterior` with empty `data`")

        self.dim = X.shape[-1]
        assert isinstance(self.dim, int)

        self.n_ensemble = 20**self.dim
        self.n_approx = n_approx
        self.observation_noise = tf.convert_to_tensor(observation_noise, tf.float64)

        assert self.n_approx > 0 and self.dim > 0 and observation_noise >= 0

        weights = tf.convert_to_tensor(
            [
                moo.sample_scalarization_weights(self.dim)
                for _ in range(self.n_ensemble)
            ],
            tf.float64,
        )
        llikelihoods = [
            moo.log_likelihood_linear_utility(w, X, Y, self.observation_noise)
            for w in weights
        ]
        self.weighted_particles = WeightedParticles(weights, llikelihoods)

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        # We first sample, for each batch, `num_samples` weights.
        weights = self.weighted_particles.sample([*b, num_samples])
        assert weights.shape == [*b, num_samples, self.dim]

        # We now compute how the sampled weights lead to sample outcomes.
        samples = tf.matmul(query_points, weights, transpose_b=True)
        assert samples.shape == tf.TensorShape([*b, n, num_samples])

        # "unpack" `samples` and switch `n` and `num_samples` dimension.
        samples = tf.transpose(
            tf.expand_dims(samples, -1),
            perm=[*range(len(b)), len(b) + 1, len(b), len(b) + 2],
        )
        assert samples.shape == [*b, num_samples, n, 1]

        return samples

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape)

        samples = self.sample(query_points, self.n_approx)
        assert samples.shape == tf.TensorShape([*b, self.n_approx, n, 1])

        mean = tf.reduce_mean(samples, axis=len(b))
        var = tf.math.reduce_variance(samples, axis=len(b))

        assert mean.shape == tf.TensorShape([*b, n, 1])
        assert var.shape == tf.TensorShape([*b, n, 1])

        return mean, var

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.

        :return: The observation noise.
        """
        return self.observation_noise


class MultIndependentGPs(trieste.models.interfaces.SupportsGetObservationNoise):
    """Multi output GP, where each output is an independent GP."""

    def __init__(
        self, X: tf.Tensor, Y: tf.Tensor, search_space: trieste.space.SearchSpace
    ):
        """Creates `k` independent GPs, assuming data set has output dimension `k`"""

        self.models: list[trieste.models.interfaces.SupportsGetObservationNoise] = []
        self.output_dim = Y.shape[-1]

        assert isinstance(self.output_dim, int) and self.output_dim > 1

        # Our models are trained on standardized objectives.
        # However, when we `predict` and `sample` we return re-scaled output.
        # For this, we need to save the mean and standard deviation.
        self.means = []
        self.stds = []

        for i in range(self.output_dim):
            single_objective = tf.gather(Y, [i], axis=1)

            # Standardize the objectives, and store the statistics to scale back later.
            sca, mean, std = utils.normalize(single_objective)
            single_data = trieste.data.Dataset(X, sca)

            self.means.append(mean)
            self.stds.append(std)

            self.models.append(create_gp(single_data, search_space))

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        # Here we sample objectives from our models.
        # Note we immediately scale them back using the stored means and standard deviation.
        list_of_samples = [
            model.sample(query_points, num_samples) * std + mean
            for mean, std, model in zip(self.means, self.stds, self.models)
        ]

        # If this is false, I give up on life.
        # But I rather go that way, then have a bug caused by the following being wrong.
        assert len(list_of_samples) == self.output_dim
        for samples in list_of_samples:
            assert samples.shape == tf.TensorShape([*b, num_samples, n, 1])

        samples = tf.reshape(
            tf.concat(list_of_samples, axis=-1), (*b, num_samples, -1, self.output_dim)
        )
        assert samples.shape == tf.TensorShape([*b, num_samples, n, self.output_dim])

        return samples

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""

        b = query_points.shape[:-1]
        assert isinstance(b, tf.TensorShape)

        means_sca, vars_sca = zip(*[m.predict(query_points) for m in self.models])

        # We de-normalize our objectives.
        # The actual predicted mean is `o * std + mean`.
        # Its variance is simply the multiplication with the previous: `v * sqrt(std)`.
        mean = [
            mean_sca * std + m
            for mean_sca, std, m in zip(means_sca, self.stds, self.means)
        ]
        variance = [
            var_sca * tf.pow(std, 2) for var_sca, std in zip(vars_sca, self.stds)
        ]

        m = tf.reshape(tf.concat(mean, -1), (*b, self.output_dim))
        v = tf.reshape(tf.concat(variance, -1), (*b, self.output_dim))

        return m, v

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.
        """
        # We simply ask the observation noise of our individual GPs.
        # Note that we do re-scale it by multiplying with their variance.
        noise = [
            m.get_observation_noise() * tf.pow(s, 2)
            for m, s in zip(self.models, self.stds)
        ]

        return tf.squeeze(tf.concat(noise, -1))


class CompositeGP(trieste.models.interfaces.SupportsGetObservationNoise):
    """Note `SupportsGetObservationNoise` is a `ProbabilisticModel`."""

    def __init__(
        self,
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        scalarization_weights: list[float],
    ):
        """Composite GP over multiple objectives given the `scalarization_weights`.

        Initiates individual GPs, one for each objective, and implements Trieste's
        API for probabilistic models. In particular, `sample` is done by sampling
        from the individual GPs and then computing the cost given the `scalarization_weights`.
        """
        self.scalarization_weights = tf.convert_to_tensor(
            scalarization_weights, tf.float64
        )
        self.o_dim = len(scalarization_weights)

        assert isinstance(data.query_points, tf.Tensor)
        assert isinstance(data.observations, tf.Tensor)

        self.models = MultIndependentGPs(
            data.query_points, data.observations, search_space
        )

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        # Here we sample objectives from our models.
        # Note we immediately scale them back using the stored means and standard deviation.
        obj_samples = self.models.sample(query_points, num_samples)
        assert obj_samples.shape == tf.TensorShape((*b, num_samples, n, self.o_dim))

        # XXX: We assume utility function has no noise.
        cost = moo.scalarize_objectives(
            tf.reshape(obj_samples, (-1, self.o_dim)), self.scalarization_weights
        )
        assert cost.shape == [prod([*b, num_samples, n]), 1]

        return tf.reshape(cost, (*b, num_samples, n, 1))

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""

        b = query_points.shape[:-1]
        assert isinstance(b, tf.TensorShape)

        o_m, o_v = self.models.predict(query_points)

        assert o_m.shape == tf.TensorShape((*b, self.o_dim))
        assert o_v.shape == tf.TensorShape((*b, self.o_dim))

        # Here we transform our predicted means and variance.
        # In particular, we want to predict the cost's mean and variance:

        # Given `X_i ~ N(m_i, v_i)`, we have:
        # `c * X_i   ~ N(c * m_i, c ** 2 * v_i)`
        # `X_i + X_j ~ N(m_i + m_j, v_i + v_j)`

        # Which together makes:
        # `c_i * X_i + c_j * X_j ~ N(c_i * m_i + c_j * m_j, c_i ** 2 * v_i + c_j ** 2 * v_j)`

        # As in: the mean is sum(w_i * m_i) and the variance is sum(w_i ** 2 * s_i)
        # We implement this with tensor operations.
        m = tf.matmul(
            tf.reshape(o_m, (-1, self.o_dim)),
            tf.reshape(self.scalarization_weights, (-1, 1)),
        )
        # XXX: we assume here that the utility function is noise free.
        v = tf.matmul(
            tf.reshape(o_v, (-1, self.o_dim)),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )

        assert m.shape == tf.TensorShape([*b, 1]) and v.shape == tf.TensorShape([*b, 1])
        return m, v

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        :return: The observation noise.
        """
        # Here we combine the observation noise of our individual GPs.
        o_noise = self.models.get_observation_noise()

        assert o_noise.shape == tf.TensorShape([self.o_dim])

        # And then we take the linear combination.
        # We follow the following math:
        #   `Var[X + Y] = Var[X] + Var[Y]`    if X and Y are independent.
        #   `Var[c * X] = c^2 * Var[X]`
        # To get:
        # `Var[U] = Var[w_1 * O_1 + ... w_k * O_k] = w_1^2 * O_1 + ... + w_k^2 O_k`
        # XXX: we assume here that the utility function is noise free.
        combined_noise = tf.matmul(
            tf.reshape(o_noise, (1, -1)),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )
        assert combined_noise.shape == tf.TensorShape([1, 1])

        return tf.squeeze(combined_noise)


class UtilityDistribution(trieste.models.interfaces.SupportsGetObservationNoise):
    """Note `SupportsGetObservationNoise` is a `ProbabilisticModel`."""

    def __init__(self, O: tf.Tensor, Y: tf.Tensor, objectives):
        """A distribution over the utility given known objective functions.

        This class implements the `Trieste` model interface(s) to represent
        a (posterior) distribution over the utility due to unknown weights.

        - The objective function is assumed known (`objectives`).
        - The utility function is assumed to be linear, and the prior over the weights is uniform.
        - `data` is supposed to contain o -> u, from which we then infer the weights.
        """
        o_dim = O.shape[-1]
        assert isinstance(o_dim, int)

        self.o_dim = o_dim
        self.objectives = objectives
        self.weight_posterior = LinearPosterior(O, Y)

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        o = self.objectives(query_points)
        assert o.shape == (*b, n, self.o_dim)

        samples = self.weight_posterior.sample(o, num_samples)

        assert samples.shape == tf.TensorShape([*b, num_samples, n, 1])
        return samples

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""
        o = self.objectives(query_points)
        predictions = self.weight_posterior.predict(o)

        return predictions

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.

        :return: The observation noise.
        """
        # There is no observation noise over the objectives:
        # we assume they are known.

        # So the only observation noise is that of the utility function:
        return self.weight_posterior.observation_noise


class MOOPosterior(trieste.models.interfaces.SupportsGetObservationNoise):
    """Note `SupportsGetObservationNoise` is a `ProbabilisticModel`."""

    def __init__(
        self,
        X: tf.Tensor,
        O: tf.Tensor,
        Y: tf.Tensor,
        search_space: trieste.space.SearchSpace,
        n_predict_samples: int = 100,
    ):
        """Creates a joint distribution from posterior over objectives and utility.

        :n_predict_samples: number of samples to approximate mean and variance.
        """
        o_dim = O.shape[-1]
        assert isinstance(o_dim, int)

        self.n_predict_samples = n_predict_samples
        self.o_dim = o_dim

        self.weight_posterior = LinearPosterior(O, Y)
        self.objectives_posterior = MultIndependentGPs(X, O, search_space)

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        # Sample `num_samples` objectives and weights, and flatten them.
        o_samples = self.objectives_posterior.sample(query_points, num_samples)
        assert o_samples.shape == tf.TensorShape([*b, num_samples, n, self.o_dim])
        o_samples = tf.reshape(o_samples, (prod([*b, num_samples]), n, self.o_dim))

        samples = self.weight_posterior.sample(o_samples, 1)
        assert samples.shape == [prod([*b, num_samples]), 1, n, 1]

        return tf.reshape(samples, (*b, num_samples, n, 1))

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        predictions = self.sample(query_points, self.n_predict_samples)
        assert predictions.shape == tf.TensorShape([*b, self.n_predict_samples, n, 1])

        mean = tf.reduce_mean(predictions, axis=len(b))
        variance = tf.math.reduce_variance(predictions, axis=len(b))

        assert mean.shape == tf.TensorShape([*b, n, 1])
        assert variance.shape == tf.TensorShape([*b, n, 1])

        return mean, variance

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.

        The observation noise is a result of the observation noise of
        the underlying objective and utility posteriors.

        :return: shaped [1]
        """
        #   `Var_e[U] = Var_e[W_1 * O_1 + ... + W_k * O_k + e]`
        #            `= Var_e[W_1 * O_1] + ... + Var_e[W_k * O_k] + Var_e[e]`

        # I personally have no idea to calculate `Var_e[W_i O_i]` so, instead,
        # I take a point estimate (MAP) of w
        w_map = self.weight_posterior.weighted_particles.map()
        assert w_map.shape == (self.o_dim,)

        # Then, assuming `w_i`l is constant, continuing from above:
        #   `Var_e[w_1 * O_1] + ... + Var_e[w_k * O_k]`
        #   `= w_i^2 Var_e[O_1] + ... + w_k^2 Var_e[O_k] + Var[e]`

        # Recall that the variance of `O`, `Var_e[O_i]`, is their observation noise:
        o_noises = self.objectives_posterior.get_observation_noise()
        assert o_noises.shape == (self.o_dim,)

        noise = tf.tensordot(o_noises, tf.pow(w_map, 2), 1)
        assert noise.shape == ()

        return noise + self.weight_posterior.get_observation_noise()
