# Testing functions for the file NoiseAveraging.py

import pytest
import numpy as np

import NoiseAveraging

def test_noise_doubling():
    import numpy as np

    orig = np.arange(0, 11, 2)
    interleave = np.arange(1, 10, 2)
    full = np.arange(0, 11, 1)
    new_samples, full_sampling = NoiseAveraging.noise_doubling(orig)
    assert np.array_equal(new_samples, interleave)
    assert np.array_equal(full, full_sampling)