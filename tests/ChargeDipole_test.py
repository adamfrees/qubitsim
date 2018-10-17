# Testing code for the Hybrid Qubit class

import math
import pytest
import numpy as np

from context import qubitsim
import qubitsim.qubit.ChargeDipole as cd

def test_qubit_initialization():
    """
    Tests for correct initialization of the qubit
    """
    qubit = cd.ChargeDipole(10.0, 10.0)
    assert qubit.ed == 10.0
    assert qubit.delta == 10.0


def test_qubit_hamiltonian_lab():
    """
    Tests correct calculation of the Hamiltonian in the lab frame
    """
    qubit = cd.ChargeDipole(20.0, 10.0)
    Hlab = qubit.hamiltonian_lab() / (2 * math.pi)
    assert (-2 * Hlab[0, 0]) == (qubit.ed)
    assert qubit.delta == Hlab[1, 0]


def test_eigenbasis_normalization():
    """
    Tests if the eigenbasis is normalized
    """
    qubit = cd.ChargeDipole(30.0, 10.0)
    for vector in qubit.qubit_basis().T:
        assert np.linalg.norm(vector) - 1 <= 1e-15