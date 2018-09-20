# Testing functions for the file NoiseAveraging.py

import pytest
import math
import numpy as np
import os

from qubitsim import NoiseAveraging

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


def test_two_sigma_doubling():
    sigma = 1.1
    original = np.linspace(-6*sigma, 6*sigma, 21)
    new_samples = NoiseAveraging.two_sigma_doubling(original, sigma)
    test_array = np.array([-6.6, -5.94, -5.28, -4.62, -3.96, -3.3, -2.64, 
    -1.98, -1.65, -1.32, -0.99, -0.66, -0.33,
    0.0, 0.33, 0.66, 0.99, 1.32, 1.65, 1.98,
    2.64, 3.3, 3.96, 4.62, 5.28, 5.94, 6.6], dtype=float)
    assert len(new_samples) == len(test_array)
    assert np.allclose(new_samples, test_array)


# def test_wing_doubling():
    sigma = 1.1
    original = np.linspace(-6*sigma, 6*sigma, 11)
    new_samples = NoiseAveraging.wing_doubling(original, sigma)
    test_array = np.array([-6.6, (-6.6-5.28)/2, -5.28, (-5.28-3.96)/2,
                           -3.96, (-3.96-2.64)/2, -2.64,
                           -1.32, 0.0, 1.32,
                           2.64, (2.64+3.96)/2, 3.96, (3.96+5.28)/2,
                           5.28, (5.28+6.6)/2, 6.6])
    print(test_array)
    print(new_samples)
    assert np.allclose(test_array, new_samples)