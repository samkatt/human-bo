"""Tests functionality of `human_bo.core`"""

import random

import torch

from human_bo import core


def generate_random_bound() -> tuple[float, float]:
    """Utility function to generate a random bound"""

    # generate two floats in [-2., 2] (random, reasonable, range of bounds to have).
    x1 = (random.random() - 0.5) * random.randint(1, 4)
    x2 = (random.random() - 0.5) * random.randint(1, 4)

    # return them in the correct order.
    return min(x1, x2), max(x1, x2)


def test_random_query():
    """Tests `core.random_query`"""
    for _ in range(10):
        n_dim = random.randint(1, 8)
        bounds = [generate_random_bound() for _ in range(n_dim)]
        queries = core.random_queries(bounds, n=2)

        assert queries.shape == (
            2,
            n_dim,
        ), "core.random_query should produce `n` queries of the size of bound."

        assert not torch.allclose(
            queries[0], queries[1]
        ), "core.random_query should produce different queries."

        for q in queries:
            for b_i, x_i in zip(bounds, q):
                assert (
                    b_i[0] < x_i < b_i[1]
                ), "core.random_query should produce queries within the bounds."
