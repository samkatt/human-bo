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


class CompositeGP(trieste.models.interfaces.ProbabilisticModel):
    def __init__(
        self,
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        scalarization_weights: list[float],
        num_approx_samples: int = 100,
    ):
        """Composite GP over multiple objectives given the `scalarization_weights`.

        Initiates individual GPs, one for each objective, and implements Trieste's
        API for probabilistic models. In particular, `sample` is done by sampling
        from the individual GPs and then computing the cost given the `scalarization_weights`.

        - `num_approx_samples` are the number of samples used approximate `predict`.
        """

        self.num_approx_samples = num_approx_samples
        self.scalarization_weights = tf.convert_to_tensor(
            scalarization_weights, tf.float64
        )

        self.models: list[trieste.models.interfaces.ProbabilisticModel] = []

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
        samples = self.sample(query_points, self.num_approx_samples)
        sample_mean = tf.reduce_mean(samples, axis=0)
        sample_var = tf.math.reduce_variance(samples, axis=0)

        return sample_mean, sample_var

    def log(self, dataset: trieste.data.Dataset | None = None) -> None:
        del dataset
        pass
