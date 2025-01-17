"""Tests functionality of `human_bo.utils`"""

import pytest

from human_bo import utils


def test_exit_if_exists():
    """Tests `core.exit_if_exists`"""
    with pytest.raises(ValueError):
        utils.exit_if_exists("pyproject.toml")
    with pytest.raises(ValueError):
        utils.exit_if_exists("scripts")

    utils.exit_if_exists("this-file-does-not-exist")
