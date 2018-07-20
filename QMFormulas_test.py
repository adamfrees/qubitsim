# Test functions for QMFormulas.py

import pytest
import numpy as np

import QMFormulas

def test_eigvector_phase_sort_test():
    test_matrix = np.array([[-1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
    new_matrix = QMFormulas.eigvector_phase_sort(test_matrix)
    result_matrix = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
    assert np.array_equal(result_matrix, new_matrix)


def test_gaussian_mean_sd_testing():
    mean_intended = 2.0
    sigma_intended = 1.0
    x_array = np.linspace(-10 * sigma_intended, 10 * sigma_intended, 201)
    gaussian_array = QMFormulas.gaussian(x_array, mean_intended, sigma_intended)
    assert np.abs(mean_intended- np.mean(gaussian_array)) < 1e-8
    assert np.abs(sigma_intended - np.std(gaussian_array)) < 1e-8