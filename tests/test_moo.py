"""Tests functionality of `human_bo.moo`"""

import random

import pytest
import tensorflow as tf

from human_bo import moo


def test_sample_scalarization_weights():
    """Tests `moo.sample_scalarization_weights()`"""

    o_dim = random.sample(range(3, 8), 1)[0]
    weights = moo.sample_scalarization_weights(o_dim)

    assert len(weights) == o_dim
    assert sum(weights) == pytest.approx(1)

    with pytest.raises(Exception):
        moo.sample_scalarization_weights(1)


def test_compute_scalarization():
    o = tf.convert_to_tensor([[0.2, 0.5], [-0.2, 0]])
    w = tf.convert_to_tensor([0.4, 0.6])

    u = moo.scalarize_objectives(o, w)

    assert tf.reduce_all(
        tf.experimental.numpy.isclose(u, tf.convert_to_tensor([[0.38], [-0.08]]))
    )


def test_log_likelihood_linear_utility():
    w = tf.constant([0.1, 0.9], tf.float64)
    o = tf.constant([[0.3, 0.4], [0.6, 0.2]], tf.float64)
    u = tf.constant([0.39, 0.24], tf.float64)
    u_noise = tf.add(u, 0.1)

    var_small = tf.constant(0.01, tf.float64)

    assert tf.reduce_all(
        tf.math.greater(
            moo.log_likelihood_linear_utility(w, o, u, var_small),
            moo.log_likelihood_linear_utility(w, o, u),
        )
    )

    assert tf.reduce_all(
        tf.math.greater(
            moo.log_likelihood_linear_utility(w, o, u, var_small),
            moo.log_likelihood_linear_utility(w, o, u_noise),
        )
    )
