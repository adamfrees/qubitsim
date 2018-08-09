# Test suite for CJ fidelity calculations

import pytest
import numpy as np

import CJFidelities as CJ


def test_chi0_initiliazation():
    H0 = np.identity(3)
    Hnoise = np.zeros((3,3))
    indices = [0, 1]
    test_matrix = np.zeros((9, 9), dtype=complex)
    test_matrix[0, 0] += 0.5
    test_matrix[0, 4] += 0.5
    test_matrix[4, 0] += 0.5
    test_matrix[4, 4] += 0.5
    cj_test = CJ.CJ(indices, H0, Hnoise)
    chi_test = cj_test.chi0
    assert np.array_equal(test_matrix, chi_test)