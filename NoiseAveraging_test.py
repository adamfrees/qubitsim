# Testing functions for the file NoiseAveraging.py

import pytest
import numpy as np

import NoiseAveraging

def noise_doubling_testing_doubling():
    original_test = np.arange(0, 12, 2)
    new_samples, full_sampling = NoiseAveraging.noise_doubling(original_test)
    assert new_samples == np.arange(1, 11, 2)


def noise_doubling_full_array_test():
    original_test = np.arange(0, 12, 2)
    new_samples, full_sampling = NoiseAveraging.noise_doubling(original_test)
    assert full_sampling == np.arange(0, 12, 1)