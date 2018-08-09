# Test suite for CJ fidelity calculations

import pytest
import numpy as np

import CJFidelities as CJ


def test_dimension_chi0():
    H0 = np.identity(3)
    Hnoise = np.zeros((3,3))
    indices = [0, 1]
    cj_test = CJ.CJ(indices, H0, Hnoise)
    chi_test = cj_test.chi0
    assert chi_test.shape == (9, 9)


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


def test_kernel_basic():
    H0 = np.array([[0, 1], [1, 0]])
    Hnoise = np.zeros((2,2))
    indices = [0, 1]
    cj_test = CJ.CJ(indices, H0, Hnoise)
    kernel_from_class = cj_test.kernel
    test_kernel = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0]])
    assert np.array_equal(kernel_from_class, test_kernel)