# Quantum Mechanics Formulas
# A selection of useful quantum mechanics base formulas
# Cameron King
# University of Wisconsin, Madison

import math

import numpy as np
import scipy.linalg as LA


def eigvector_phase_sort(eig_matrix):
    for i in range(eig_matrix.shape[1]):
        if eig_matrix[0, i] < 0:
            eig_matrix[:, i] *= -1
    return eig_matrix

def gaussian(x, mean, sigma):
    """Return the value of a normalized gaussian probability distribution
    with mean mu, standard deviation sigma, at the value x"""
    return np.exp(-np.square(x-mean)/(2*np.square(sigma))) / (np.sqrt(2*np.pi*sigma**2))


def basischange(rho0, U):
    """Perform a matrix transformation into the 
    basis defined by U. Can also be used for unitary 
    transformation of a density matrix rho0"""
    return U @ rho0 @ U.conj().T


def processFidelity(chiIdeal, chiActual):
    """Calculate the process fidelity between 
    two process matrices chiIdeal and chiActual. 
    chiIdeal and chiActual are not assumed to be unitary 
    processes"""
    trace1 = np.real(np.trace(chiIdeal @ chiActual))
    # trace2 = np.sqrt(np.real(np.trace(chiIdeal @ chiIdeal)))
    # trace3 = np.sqrt(np.real(np.trace(chiActual @ chiActual)))
    return trace1


def processInfidelity(chiIdeal, chiActual):
    """Calculate the process infidelity between two 
    matrices chiIdeal and chiActual. chiIdeal and 
    chiActual are not assumed to be unitary processes."""
    return 1 - processFidelity(chiIdeal, chiActual)


def processInfidelityUnitary(chiIdeal, chiActual):
    """Calculate the process fidelity assuming 
    unitary processes"""
    return 1 - np.real(np.trace(chiIdeal @ chiActual))


def commutator(A, B):
    """Return the commutator between two equivalently dimensioned 
    matrices A and B"""
    return A @ B - B @ A


def anticommutator(A, B):
    """Return the anti-commutator between two equivalently dimensioned
    matrices A and B"""
    return A @ B + B @ A


def frobenius_inner(A, B):
    """Return the Frobenius norm between two equivalently dimensioned
    matrices A and B"""
    test1 = np.sqrt(np.abs(np.trace(np.dot(A.conj().T, A.T))))
    test2 = np.sqrt(np.abs(np.trace(np.dot(B.conj().T, B))))
    test3 = np.abs(np.trace(np.dot(A.conj().T, B)))
    return np.sqrt(test3 / (test1 * test2))


def derivative(func, test_point, order):
    h = 0.1
    h = 1e-10
    # test_array = test_point + np.arange(-4*h, 5*h, h)
    # test_array = test_point + np.arange(-2*h, 3*h, h)
    if order == 0:
        return func(test_point)
    elif order == 1:
        test_array = test_point + np.arange(-4*h, 5*h, h)
        eval_array = func(test_array)
        coeff_array = np.array([1.0 / 280.0, -4.0/105.0, 0.2, -0.8, 0.0, 0.8, -0.2, -4.0/105.0, -1.0/280.0])
        # coeff_array = np.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])
        return np.dot(coeff_array, eval_array) / h
    elif order == 2:
        test_array = test_point + np.arange(-4*h, 5*h, h)
        eval_array = func(test_array)
        coeff_array = np.array([-1.0/560.0, 8.0/315.0, -0.25, 1.6, -205.0/72.0, 1.6, -0.2, 8.0/315.0, -1.0/560.0])
        # coeff_array = np.array([-1.0/12.0, 4.0/3.0, -2.5, 4.0/3.0, -1.0/12.0])
        return np.dot(coeff_array, eval_array) / h**2
    elif order == 3:
        # coeff_array = np.array([-7.0/240.0, 0.3, -169.0/120.0, 61.0/30.0, 0, -61.0/30.0, 169.0/120.0, -0.3, 7.0/240.0])
        coeff_array = np.array([-0.5, 1.0, 0.0, -1.0, 0.5])
        return np.dot(coeff_array, eval_array) / h**3
    else:
        return "Error in order parameter"