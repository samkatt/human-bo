"""Tests functionality of `human_bo.moo`"""

import random

import pytest

from human_bo import moo


def test_sample_preference_weights():
    """Tests `moo.sample_preference_weights()`"""

    o_dim = random.sample(range(3, 8), 1)[0]
    weights = moo.sample_preference_weights(o_dim)

    assert len(weights) == o_dim
    assert sum(weights) == pytest.approx(1)

    with pytest.raises(Exception):
        moo.sample_preference_weights(1)
