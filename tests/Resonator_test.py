# Testing code for the resonator class

import pytest
import math
import numpy as np

from context import qubitsim
import qubitsim.resonator as Resonator

def test_number_operator():
    omega = 10.0
    dim = 3
    res = Resonator.Resonator(omega, dim)
    test_number = np.diag(np.array([0, 1, 2]))
    assert np.array_equal(res.number_operator(), test_number)

def test_creation_operator():
    omega = 10.0
    dim = 4
    res = Resonator.Resonator(omega, dim)
    test_adag = np.array([[0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, math.sqrt(2), 0, 0],
                          [0, 0, math.sqrt(3), 0]])
    assert np.array_equal(res.creation_operator(), test_adag)


def test_annihilation_operator():
    omega = 10.0
    dim = 4
    res = Resonator.Resonator(omega, dim)
    test_a = np.array([[0, 1, 0, 0],
                          [0, 0, math.sqrt(2), 0],
                          [0, 0, 0, math.sqrt(3)],
                          [0, 0, 0, 0]])
    assert np.array_equal(res.annihilation_operator(), test_a)


def test_commutation_relation_creation_annihilation():
    omega = 10.0
    dim = 5
    res = Resonator.Resonator(omega, dim)
    a = res.annihilation_operator()
    adag = res.creation_operator()
    identity = np.eye(dim, dtype=float)
    identity[4, 4] = -(dim-1)
    test_identity = (a @ adag) - (adag @ a)
    assert np.allclose(test_identity, identity)
