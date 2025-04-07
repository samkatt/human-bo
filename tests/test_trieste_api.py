"""Tests functionality of `human_bo.trieste_api`"""

import numpy as np
import pytest
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


def test_compute_utility():
    """Test `trieste_api.compute_utility`."""
    o = tf.convert_to_tensor([[0.2, 0.5], [-0.2, 0]])
    w = tf.convert_to_tensor([0.4, 0.6])

    u = trieste_api.compute_utility(o, w)

    assert np.array(u) == pytest.approx(np.array([[0.38], [-0.08]]))


def test_create_trieste_observer():
    """Test `trieste_api.create_trieste_observer`."""
    f = trieste_api.create_trieste_test_function("Zhou")
    o = trieste_api.create_trieste_observer(f.objective, None)
    o_noise = trieste_api.create_trieste_observer(f.objective, [0.1])

    x = f.search_space.sample(4)

    y = f.objective(x)

    data_no_noise = o(x)
    data_noise = o_noise(x)

    assert isinstance(data_no_noise, trieste.data.Dataset)
    assert isinstance(data_noise, trieste.data.Dataset)

    tf.assert_equal(y, data_no_noise.observations)
    tf.debugging.assert_none_equal(y, data_noise.observations)
