"""Tests functionality of `human_bo.trieste_api`"""

import tensorflow as tf
import trieste

from human_bo import trieste_api


def test_create_trieste_test_function():
    """Tests `trieste_api.create_trieste_test_function`."""
    for f in ["Zhou", "Forrester", "Levy1D"]:
        p = trieste_api.create_trieste_test_function(f)
        assert isinstance(
            p, trieste.objectives.single_objectives.SingleObjectiveTestProblem
        )

        ob = trieste.objectives.utils.mk_observer(p.objective)
        assert isinstance(ob(p.search_space.sample_sobol(5)), trieste.data.Dataset)

        for obs in ob(p.minimizers).observations:
            assert tf.experimental.numpy.isclose(obs, p.minimum, atol=0.01)


def test_create_moo_trieste_test_function():
    """Tests `trieste_api.create_trieste_test_function` on MOO functions."""

    x_dim = 7
    o_dim = 4
    p = trieste_api.create_trieste_test_function("DTLZ2", x_dim, o_dim)
    assert p.dim == x_dim

    y = p.objective(p.search_space.sample(1))
    assert y.shape == (1, o_dim)
