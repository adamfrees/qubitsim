# A class that implements state evolution for
# the purpose of calculating infidelities using Choi-Jamilkowski
# matrices

import numpy as np
import scipy.linalg as LA

class CJ (object):
    """
    Return a Choi-Jamilkowski matrix after a given amount
    of state evolution. This is equivalent to the chi-matrix
    for the evolution
    """
    def __init__(self, indices, hamiltonian):
        """
        Initialize a Choi-Jamilkowski instance with the subspace
        of interest given by indices, and the kernel of the unitary
        evolution, given by hamiltonian (units: angular GHz).

        If non-Hamiltonian evolution is needed go elsewhereself.
        """
        dim = hamiltonian.shape[0]
        norm = 1.0 / float(len(indices))
        converted_indices = int((dim + 1) * indices)
        chi0 = np.zeros((dim**2, dim**2), dtype=complex)
        self.chi0 = chi0[np.ix_(converted_indices, converted_indices)] = norm
        self.kernel = np.kron(np.identity(hamiltonian.shape[0]), hamiltonian)

    def chi_final(self, tfinal):
        """
        Using the kernel given in initialition, find the final chi_matrix
        """
        unitary = LA.expm(-1j*tfinal*self.kernel)
        return unitary @ self.chi0 @ unitary.conj().T

    def infidelity(self, tfinal):
        """
        Calculate the infidelity of the operation being characterized.
        """
        chi_f = self.chi_final(tfinal)
        trace1 = np.real(np.trace(self.chi0 @ chi_f))
        trace2 = np.real(np.trace(chi_f))
        trace3 = np.real(np.trace(self.chi0))
        return 1 - trace1 / (trace2 * trace3)
