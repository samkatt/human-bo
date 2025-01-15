"""Tests functionality of `human_bo.test_functions`"""

import torch

from human_bo import test_functions


def test_sample_initial_points():
    """Tests test_functions.sample_initial_points"""

    def f(x):
        return x[:, 0] - 0.34 * x[:, 1] + torch.normal(0, 0.2, [len(x)])

    bounds = [(-0.25, 2.3), (1.3, 10.4)]
    n_init = 4

    x_samples, y_samples = test_functions.sample_initial_points(f, bounds, n_init)

    assert (
        len(x_samples) == len(y_samples) == n_init
    ), "`sample_initial_points` Number of x and y samples should be `n_init`."

    for x, y in zip(x_samples, y_samples):
        assert not torch.allclose(
            y, f(x.unsqueeze(0))
        ), "`sample_initial_points` output be stochastic."

    for x in x_samples:
        for b, x_i in zip(bounds, x):
            assert (
                b[0] < x_i < b[1]
            ), "`sample_initial_points` x should be sampled within the bounds."
