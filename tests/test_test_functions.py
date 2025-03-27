"""Tests functionality of `human_bo.test_functions`"""

import pytest
import tensorflow as tf
import torch
import trieste

from human_bo import test_functions


def test_forrester():
    """Tests `test_functions.Forrester`"""
    p = test_functions.ForresterBotorch()
    assert p.optimal_value == pytest.approx(6.020738786441099)


def test_create_moo_function():
    """Tests `test_functions.pick_moo_test_function`"""
    p = test_functions.create_moo_test_function("BraninCurrin", [0.2, 0.45])
    p(torch.rand([4, 2]))


def test_create_trieste_test_function():
    """Tests `test_functions.create_trieste_test_function`"""
    for f in ["Zhou", "Forrester", "Levy1D"]:
        p = test_functions.create_trieste_test_function(f)
        assert isinstance(
            p, trieste.objectives.single_objectives.SingleObjectiveTestProblem
        )

        ob = trieste.objectives.utils.mk_observer(p.objective)
        assert isinstance(ob(p.search_space.sample_sobol(5)), trieste.data.Dataset)

        for obs in ob(p.minimizers).observations:
            assert tf.experimental.numpy.isclose(obs, p.minimum, atol=0.01)
