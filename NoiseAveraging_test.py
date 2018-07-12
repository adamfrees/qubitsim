# Testing functions for the file NoiseAveraging.py

import pytest
import numpy as np

import NoiseAveraging


class Test_Doubling_function(object):
    original_test = np.arange(0, 12, 2)
    new_samples, full_sampling = NoiseAveraging.noise_doubling(original_test)

    def test_noise_doubling(self, new_samples):
        assert new_samples == np.arange(1, 11, 2)


    def test_noise_doubling_full_array(self, full_sampling):
        assert full_sampling == np.arange(0, 12, 1)