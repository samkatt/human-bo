"""Tests functionality of `human_bo.posteriors`"""

import numpy as np
import pytest

from human_bo import posteriors


def test_WeightedPF():
    particles = [10.0, 11.0, 12.0]
    log_likelihoods = [-0.3, 10.4, 3]
    n = 100

    weighted_pf = posteriors.WeightedParticles(particles, log_likelihoods)

    samples = weighted_pf.sample(n).numpy()
    assert samples.shape == (n,)

    sample_values, sample_counts = np.unique(samples, return_counts=True)
    assert len(sample_values) <= len(particles)
    assert sample_values[np.argmax(sample_counts)] == pytest.approx(11.0)

    assert weighted_pf.map().numpy() == pytest.approx(11.0)
