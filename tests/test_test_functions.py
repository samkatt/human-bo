"""Tests functionality of `human_bo.test_functions`"""

import pytest
import torch

from human_bo import test_functions


def test_forrester():
    """Tests `test_functions.Forrester`"""
    p = test_functions.Forrester()
    assert p.optimal_value == pytest.approx(6.020738786441099)


def test_create_moo_function():
    """Tests `test_functions.pick_moo_test_function`"""
    p = test_functions.pick_moo_test_function("BraninCurrin", [0.2, 0.45])
    p(torch.rand([4, 2]))
