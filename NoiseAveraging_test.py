# Testing functions for the file NoiseAveraging.py

import pytest
import math
import numpy as np

import NoiseAveraging

def test_noise_doubling():
    orig = np.arange(0, 11, 2)
    interleave = np.arange(1, 10, 2)
    full = np.arange(0, 11, 1)
    new_samples, full_sampling = NoiseAveraging.noise_doubling(orig)
    assert np.array_equal(new_samples, interleave)
    assert np.array_equal(full, full_sampling)


def test_even_area_sampling():
    from scipy.special import erf
    samples = 51
    difference = 2.0 / (samples - 1)
    sigma = 2.5
    test_samples = NoiseAveraging.even_area_sampling(samples, sigma)
    differences = np.diff(erf(test_samples / (math.sqrt(2) * sigma)))
    print(differences)
    test_total = difference * (samples-3)
    assert len(test_samples) == (samples-2)
    assert np.sum(differences) == test_total