"""Tests functionality of `human_bo.test_functions`"""

import pytest

from human_bo import test_functions


def test_forrester():
    """Tests `test_functions.Forrester`"""
    p = test_functions.Forrester()
    assert p.optimal_value == pytest.approx(6.020738786441099)
