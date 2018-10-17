# Testing code for the Hybrid Qubit class

import math
import pytest
import numpy as np

from context import qubitsim
import qubitsim.qubit.ChargeQuadrupole as cq

def test_qubit_initialization():
    """
    Tests for correct initialization of the qubit
    """
    qubit = cq.ChargeQuadrupole(10.0, 10.0, 5.0, 2.0)
    assert qubit.eq == 10.0
    assert qubit.delta1 == 10.0
    assert qubit.delta2 == 5.0
    assert qubit.ed == 2.0


def test_qubit_hamiltonian_lab():
    """
    Tests correct calculation of the Hamiltonian in the lab frame
    """
    qubit = cq.ChargeQuadrupole(20.0, 10.0, 5.0, 1.0)
    Hlab = qubit.hamiltonian_lab() / (2 * math.pi)
    assert (Hlab[1, 2]) == (qubit.ed)
    assert math.isclose(Hlab[0, 1], ((qubit.delta1 + qubit.delta2) / math.sqrt(2)))
    assert math.isclose(Hlab[0, 2], ((qubit.delta1 - qubit.delta2) / math.sqrt(2)))
    assert 2 * Hlab[0, 0] == qubit.eq


def test_eigenbasis_normalization():
    """
    Tests if the eigenbasis is normalized
    """
    qubit = cq.ChargeQuadrupole(30.0, 10.0, 10.0, 0.0)
    for vector in qubit.qubit_basis().T:
        assert np.linalg.norm(vector) - 1 <= 1e-15


def test_ideal_CQ_initialization():
    """
    Tests if the ideal case qubit is correct
    """
    qubit1 = cq.ChargeQuadrupole(10.0, 5.0, 5.0, 0.0)
    qubit2 = cq.IdealCQ(10.0, 5.0)
    assert np.array_equal(qubit1.hamiltonian_lab(), qubit2.hamiltonian_lab())