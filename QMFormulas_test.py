# Test functions for QMFormulas.py

import pytest
import numpy as np

import QMFormulas

def eigvector_phase_sort_test():
    test_matrix = np.array([[-1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
    new_matrix = QMFormulas.eigvector_phase_sort(test_matrix)
    result_matrix = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
    assert np.array_equal(result_matrix, new_matrix)