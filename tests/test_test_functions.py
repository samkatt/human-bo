"""Tests functionality of `human_bo.test_functions`"""

import torch
from human_bo import test_functions


def test_sample_initial_points():
    """Tests test_functions.sample_initial_points"""

    def f(x):
        return x[:, 0] - 0.34 * x[:, 1]

    bounds = [(-0.25, 2.3), (1.3, 10.4)]
    noise = 0.2
    n_init = 4

    x_samples, y_samples, y_true = test_functions.sample_initial_points(
        f, bounds, n_init, noise
    )

    assert (
        len(x_samples) == len(y_samples) == len(y_true) == n_init
    ), "Number of x and y samples should be `n_init`."
    assert (
        y_samples.shape == y_true.shape
    ), "Shape of true and observed y should be the same."

    for x, y in zip(x_samples, y_true):
        assert torch.allclose(y, f(x[None])), "y should be f(x)."

    for x in x_samples:
        for b, x_i in zip(bounds, x):
            assert b[0] < x_i < b[1], "x should be sampled within the bounds."
