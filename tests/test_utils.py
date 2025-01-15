"""Tests functionality of `human_bo.utils`"""

from human_bo import utils
import pytest


def test_exit_if_exists():
    """Tests `core.exit_if_exists`"""
    with pytest.raises(ValueError):
        utils.exit_if_exists("setup.py")
    with pytest.raises(ValueError):
        utils.exit_if_exists("human_bo")

    utils.exit_if_exists("this-file-does-not-exist")
