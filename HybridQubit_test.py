# Testing code for the Hybrid Qubit class

import math
import pytest
import numpy as np
import HybridQubit as hybrid

def test_qubit_initialization():
    """
    Tests for correct initialization of the qubit
    """
    qubit = hybrid.HybridQubit(10.0, 10.0, 7.0, 7.0)
    assert qubit.ed == 10.0
    assert qubit.stsplitting == 10.0
    assert qubit.delta1 == 7.0
    assert qubit.delta2 == 7.0


def test_qubit_hamiltonian_lab():
    """
    Tests correct calculation of the Hamiltonian in the lab frame
    """
    qubit = hybrid.HybridQubit(20.0, 10.0, 7.0, 4.0)
    Hlab = qubit.hamiltonian_lab() / (2 * math.pi)
    assert (-2 * Hlab[0, 0]) == (qubit.ed)
    assert qubit.delta1 == Hlab[2, 0]
    assert qubit.delta2 == (-Hlab[1, 2])
    assert (-0.5 * qubit.ed + qubit.stsplitting) == Hlab[1, 1]


def test_eigenbasis_normalization():
    """
    Tests if the eigenbasis is normalized
    """
    qubit = hybrid.HybridQubit(30.0, 10.0, 7.0, 4.0)
    for vector in qubit.qubit_basis().T:
        assert np.linalg.norm(vector) - 1 <= 1e-15


def test_derivatives():
    qubit = hybrid.SOSSHybrid(2.0, 10.0)
    deriv_from_method = qubit.splitting_derivative(1)
    assert deriv_from_method < 1e-8