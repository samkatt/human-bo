"""Various models and posteriors."""

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
