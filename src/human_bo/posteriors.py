"""Various models and posteriors."""

import tensorflow as tf
import trieste


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


def create_trieste_multi_independent_gp(
    data: trieste.data.Dataset, search_space: trieste.space.SearchSpace, num_output: int
):
    """Creates multi-objective model of independent GPs."""
    assert num_output > 0

    gps: list[trieste.models.interfaces.ProbabilisticModel] = []

    for i in range(num_output):
        single_objective = tf.gather(data.observations, [i], axis=1)
        single_data = trieste.data.Dataset(data.query_points, single_objective)
        gps.append(create_trieste_gp(single_data, search_space))

    return trieste.models.interfaces.ModelStack(*[(gp, i) for i, gp in enumerate(gps)])
