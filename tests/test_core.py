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


def test_sample_initial_points():
    """Tests core.sample_initial_points"""

    def f(x):
        return x[:, 0] - 0.34 * x[:, 1] + torch.normal(0, 0.2, [len(x)])

    bounds = [(-0.25, 2.3), (1.3, 10.4)]
    n_init = 4

    xs, ys = core.sample_initial_points(f, bounds, n_init)

    assert (
        len(xs) == len(ys) == n_init
    ), "`sample_initial_points` Number of x and y samples should be `n_init`."

    for x, y in zip(xs, ys):
        assert not torch.allclose(
            y, f(x.unsqueeze(0))
        ), "`sample_initial_points` output be stochastic."

    for x in xs:
        for b, x_i in zip(bounds, x):
            assert (
                b[0] < x_i < b[1]
            ), "`sample_initial_points` x should be sampled within the bounds."

    x, y = core.sample_initial_points(f, bounds, 0)
