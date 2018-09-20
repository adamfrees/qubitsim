# Test functions for QMFormulas.py

import pytest
import numpy as np

from context import qubitsim
import qubitsim.QMFormulas as QMFormulas

def test_eigvector_phase_sort_test():
    test_matrix = np.array([[-1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
    new_matrix = QMFormulas.eigvector_phase_sort(test_matrix)
    result_matrix = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]])
    assert np.array_equal(result_matrix, new_matrix)


def test_gaussian_mean_sd_testing():
    mean_intended = 2.0
    sigma_intended = 1.0
    x_array = np.linspace(-10 * sigma_intended, 10 * sigma_intended, 201)
    gaussian_array = QMFormulas.gaussian(x_array, mean_intended, sigma_intended)
    norm = np.trapz(gaussian_array, x=x_array)
    gaussian_array /= norm
    x_avg = np.trapz(np.multiply(gaussian_array, x_array), x=x_array)
    x2_avg = np.trapz(np.multiply(np.square(x_array), gaussian_array), x=x_array)
    stdev = np.sqrt(x2_avg - np.square(x_avg))
    assert np.abs(mean_intended - x_avg) < 1e-8
    assert np.abs(sigma_intended - stdev) < 1e-8


def test_basis_change_real():
    from math import sqrt
    input_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    rot_matrix = np.array([[1, -1], [1, 1]], dtype=complex) / sqrt(2)
    output_matrix = QMFormulas.basischange(input_matrix, rot_matrix)
    test_output = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(test_output, output_matrix, rtol=1e-8, atol=1e-12)


def test_basis_change_imag():
    input_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    rot_matrix = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
    test_output = np.array([[0, -1], [-1, 0]], dtype=complex)
    output_matrix = QMFormulas.basischange(input_matrix, rot_matrix)
    assert np.allclose(test_output, output_matrix, rtol=1e-8, atol=1e-12)


def test_commutator_pauli():
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(2.j * pauli_z, QMFormulas.commutator(pauli_x, pauli_y),
                       rtol=1e-8, atol=1e-12)