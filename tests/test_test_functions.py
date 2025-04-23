"""Tests functionality of `human_bo.test_functions.`"""

import tensorflow as tf

from human_bo import test_functions


def test_brannin_currin():
    """Tests `human_bo.test_functions.currin` implementation ."""
    x = tf.convert_to_tensor([[0, 0], [0.5, 0.5], [1, 1]])

    # Got these from testing Botorch's implementation.
    y = tf.convert_to_tensor([[3.0000], [7.4051], [4.0053]])

    currin = test_functions.currin(x, tf.pow, tf.exp)

    assert tf.reduce_all(tf.experimental.numpy.isclose(y, currin))
