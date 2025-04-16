"""Various models and posteriors."""

import tensorflow as tf
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

        self.models: list[trieste.models.interfaces.SupportsGetObservationNoise] = []

        # Our models are trained on standardized objectives.
        # However, when we `predict` and `sample` we return re-scaled output.
        # For this, we need to save the mean and standard deviation.
        self.o_means = []
        self.o_stds = []

        for i in range(len(scalarization_weights)):
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

        num_queries = query_points.shape[0]

        # Here we sample objectives from our models.
        # Note we immediately scale them back using the stored means and standard deviation.
        obj_samples = [
            model.sample(query_points, num_samples) * self.o_stds[o] + self.o_means[o]
            for o, model in enumerate(self.models)
        ]

        # If this is false, I give up on life.
        # But I rather go that way, then have a bug caused by the following being wrong.
        assert len(obj_samples) == len(self.scalarization_weights)

        objs = tf.reshape(
            tf.concat(obj_samples, axis=-1), (-1, len(self.scalarization_weights))
        )

        cost = moo.scalarize_objectives(objs, self.scalarization_weights)
        return tf.reshape(cost, (num_samples, num_queries, 1))

    def predict(
        self, query_points: trieste.types.TensorType
    ) -> tuple[trieste.types.TensorType, trieste.types.TensorType]:
        """Abstract method of `ProbabilisticModel`."""

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
        mean = tf.matmul(
            tf.concat(o_means, axis=-1), tf.reshape(self.scalarization_weights, (-1, 1))
        )
        var = tf.matmul(
            tf.concat(o_vars, axis=-1),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )

        return mean, var

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        """Abstract method of `ProbabilisticModel`, unused in this code base."""
        del dataset
        pass

    def get_observation_noise(self) -> trieste.types.TensorType:
        """Abstract method of `SupportsGetObservationNoise`.

        Return the variance of observation noise.

        :return: The observation noise.
        """
        # Here we combine the observation noise of our individual GPs.
        # XXX: Not sure if this is mathematically correct!

        # We first grab (unscaled) noise of each objective.
        o_noise = [
            m.get_observation_noise() * tf.pow(v, 2)
            for m, v in zip(self.models, self.o_stds)
        ]

        # And then we take the linear combination.
        combined_noise = tf.matmul(
            tf.concat(o_noise, axis=-1),
            tf.reshape(tf.pow(self.scalarization_weights, 2), (-1, 1)),
        )

        return combined_noise
