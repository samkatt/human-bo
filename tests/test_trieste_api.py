"""Tests functionality of `human_bo.trieste_api`"""

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


def test_inverse_BraninCurrin():
    """Tests creating (inverse) Branin with `trieste_api.create_trieste_test_function`."""

    # test InverseBranin
    inverse_branin = trieste_api.create_trieste_test_function("InverseBranin")

    x = tf.convert_to_tensor([[0, 0], [0.5, 0.5], [1, 1]])

    # I got this from calling Botorch's implementation and negating the output.
    y_inverse_branin = tf.convert_to_tensor([[-308.1291], [-24.1300], [-145.8722]])

    assert tf.reduce_all(
        tf.experimental.numpy.isclose(y_inverse_branin, inverse_branin.objective(x))
    )

    # test InverseBraninCurrin
    y_inverse_currin = tf.convert_to_tensor([[-3.0000], [-7.4051], [-4.0053]])
    y_inverse_bc = tf.concat((y_inverse_branin, y_inverse_currin), axis=-1)

    bc = trieste_api.create_trieste_test_function("InverseBraninCurrin")

    assert tf.reduce_all(tf.experimental.numpy.isclose(y_inverse_bc, bc.objective(x)))


def test_compute_utility():
    """Test `trieste_api.compute_utility`."""
    o = tf.convert_to_tensor([[0.2, 0.5], [-0.2, 0]])
    w = tf.convert_to_tensor([0.4, 0.6])

    u = trieste_api.compute_utility(o, w)

    assert tf.reduce_all(
        tf.experimental.numpy.isclose(u, tf.convert_to_tensor([[0.38], [-0.08]]))
    )


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


def test_create_trieste_acqf():
    """Tests `trieste_api`.create_trieste_acqf."""
    # Test UCB will fail "gracefully" when not given a UCB beta value.

    with pytest.raises(AssertionError):
        trieste_api.create_trieste_acqf("UCB", trieste.space.Box([0], [1]), {})
