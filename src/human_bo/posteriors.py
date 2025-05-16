"""Various models and posteriors."""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from human_bo import moo, utils


def create_trieste_gp(
    data: trieste.data.Dataset, search_space: trieste.space.SearchSpace
):
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

    def sample(self, n: int) -> tf.Tensor:
        """Samples `n` `particles` according to their `logits`."""
        assert n > 0
        return tf.gather(self.particles, self.distr.sample(n))

    def map(self) -> Any:
        """Returns the most likely element in `particles` according to `logits`.

        Does *not care* for repeating particles (as in, will not add their weight).
        """
        return tf.gather(self.particles, self.distr.mode())


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
        # TODO: Reconsider whether we should be normalizing HERE.

        self.scalarization_weights = tf.convert_to_tensor(
            scalarization_weights, tf.float64
        )
        self.o_dim = len(scalarization_weights)

        self.models: list[trieste.models.interfaces.SupportsGetObservationNoise] = []

        # Our models are trained on standardized objectives.
        # However, when we `predict` and `sample` we return re-scaled output.
        # For this, we need to save the mean and standard deviation.
        self.o_means = []
        self.o_stds = []

        for i in range(self.o_dim):
            single_objective = tf.gather(data.observations, [i], axis=1)

            # Standardize the objectives, and store the statistics to scale back later.
            sca, mean, std = utils.normalize(single_objective)
            single_data = trieste.data.Dataset(data.query_points, sca)

            self.o_means.append(mean)
            self.o_stds.append(std)

            self.models.append(create_trieste_gp(single_data, search_space))

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        # Here we sample objectives from our models.
        # Note we immediately scale them back using the stored means and standard deviation.
        obj_samples = [
            model.sample(query_points, num_samples) * self.o_stds[o] + self.o_means[o]
            for o, model in enumerate(self.models)
        ]

        # If this is false, I give up on life.
        # But I rather go that way, then have a bug caused by the following being wrong.
        assert len(obj_samples) == len(self.scalarization_weights)
        for samples in obj_samples:
            assert samples.shape == tf.TensorShape([*b, num_samples, n, 1])

        objs = tf.reshape(
            tf.concat(obj_samples, axis=-1), (-1, len(self.scalarization_weights))
        )
        assert objs.shape == tf.TensorShape([*b, num_samples * n, self.o_dim])

        # XXX: We assume utility function has no noise.
        cost = moo.scalarize_objectives(objs, self.scalarization_weights)
        assert cost.shape == tf.TensorShape([*b, num_samples * n, 1])

        return tf.reshape(cost, (*b, num_samples, n, 1))

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""

        b = query_points.shape[:-1]
        assert isinstance(b, tf.TensorShape)

        o_means_sca, o_vars_sca = zip(*[m.predict(query_points) for m in self.models])

        # Here we de-normalize our objectives.
        # The actual predicted mean is `o * std + mean`.
        # Its variance is simply the multiplication with the previous: `v * sqrt(std)`.
        o_means = [
            o * std + m for o, std, m in zip(o_means_sca, self.o_stds, self.o_means)
        ]
        o_vars = [v * tf.pow(std, 2) for v, std in zip(o_vars_sca, self.o_stds)]

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
            tf.concat(o_means, axis=-1), tf.reshape(self.scalarization_weights, (-1, 1))
        )
        # XXX: we assume here that the utility function is noise free.
        v = tf.matmul(
            tf.concat(o_vars, axis=-1),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )

        assert m.shape == tf.TensorShape([*b, 1]) and v.shape == tf.TensorShape([*b, 1])
        return m, v

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.

        :return: The observation noise.
        """
        # Here we combine the observation noise of our individual GPs.
        # We first grab (unscaled) noise of each objective.
        o_noise = [
            m.get_observation_noise() * tf.pow(s, 2)
            for m, s in zip(self.models, self.o_stds)
        ]

        # And then we take the linear combination.
        # XXX: we assume here that the utility function is noise free.
        combined_noise = tf.matmul(
            tf.concat(o_noise, axis=-1),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )
        assert combined_noise.shape == tf.TensorShape([1, 1])

        return tf.squeeze(combined_noise)


class UtilityDistribution(trieste.models.interfaces.SupportsGetObservationNoise):
    """Note `SupportsGetObservationNoise` is a `ProbabilisticModel`."""

    def __init__(
        self,
        data: trieste.data.Dataset,
        objectives,
    ):
        """A distribution over the utility given known objective functions.

        This class implements the `Trieste` model interface(s) to represent
        a (posterior) distribution over the utility due to unknown weights.

        - The objective function is assumed known (`objectives`).
        - The utility function is assumed to be linear, and the prior over the weights is uniform.
        - `data` is supposed to contain o -> u, from which we then infer the weights.
        """
        if tf.math.count_nonzero(data.query_points) == 0:
            raise ValueError("Cannot initiate `UtilityDistribution` with empty `data`")

        o = data.query_points
        u = data.observations
        n, o_dim = o.shape

        assert isinstance(o, tf.Tensor) and isinstance(u, tf.Tensor)
        assert isinstance(n, int) and isinstance(o_dim, int)

        # TODO: make this input?
        self.utility_noise = tf.convert_to_tensor(0.1, tf.float64)
        self.o_dim = o_dim

        self.objectives = objectives
        self.n_particles = 20**o_dim  # number of weight samples.
        self.n_predictions = 100  # number of samples used to `self.predict`.

        weights = tf.convert_to_tensor(
            [moo.sample_scalarization_weights(o_dim) for _ in range(self.n_particles)],
            tf.float64,
        )
        llikelihoods = [
            moo.log_likelihood_linear_utility(w, o, u, self.utility_noise)
            for w in weights
        ]
        self.weighted_particles = WeightedParticles(weights, llikelihoods)

    def sample(
        self, query_points: trieste.types.TensorType, num_samples: int
    ) -> trieste.types.TensorType:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape) and isinstance(n, int)

        o = self.objectives(query_points)
        weights = self.weighted_particles.sample(num_samples)

        samples = tf.transpose(tf.matmul(o, weights, transpose_b=True))
        assert samples.shape == tf.TensorShape([*b, num_samples, n])

        return tf.expand_dims(samples, axis=-1)

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""
        b, n = query_points.shape[:-2], query_points.shape[-2]
        assert isinstance(b, tf.TensorShape)

        samples = self.sample(query_points, self.n_predictions)
        assert samples.shape == tf.TensorShape([*b, self.n_predictions, n, 1])

        mean = tf.reduce_mean(samples, axis=len(b))
        var = tf.math.reduce_variance(samples, len(b))

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
        return self.utility_noise
