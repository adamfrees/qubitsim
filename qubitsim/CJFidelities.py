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
    def __init__(self, indices, hamiltonian, noise_hamiltonian, noise_type = 'quasistatic'):
        """
        Initialize a Choi-Jamilkowski instance with the subspace
        of interest given by indices, and the kernel of the unitary
        evolution, given by hamiltonian (units: angular GHz).

        If non-Hamiltonian evolution is needed go elsewhere.
        """
        dim = hamiltonian.shape[0]
        norm = 1.0 / float(len(indices))
        converted_indices = [(int(dim) + 1) * x for x in indices]
        chi0 = np.zeros((dim**2, dim**2), dtype=complex)
        chi0[np.ix_(converted_indices, converted_indices)] = norm
        self.chi0 = chi0
        self.noise_type = noise_type
        if noise_type == 'quasistatic':
            shifted_hamiltonian = hamiltonian + noise_hamiltonian
            shifted_energies = LA.eigh(shifted_hamiltonian)[0]
            shifted_hamiltonian = np.diag(shifted_energies)
            noise_hamiltonian = np.zeros((dim,dim))
            self.kernel = np.kron(np.identity(dim), shifted_hamiltonian)
        else:
            self.kernel = np.kron(np.identity(dim), hamiltonian)
        self.noise = np.kron(np.identity(dim), noise_hamiltonian)
        self.rot_basis = np.kron(np.identity(dim), hamiltonian)

    def chi_final(self, tfinal):
        """
        Using the kernel given in initialition, find the final chi_matrix
        """
        if tfinal == 0.0:
            return self.chi0
        else:
            unitary = LA.expm(-1j*tfinal*(self.kernel + self.noise))
            return unitary @ self.chi0 @ unitary.conj().T
            
    def chi_final_RF(self, tfinal):
        """
        Find the chi_matrix in the rotating frame defined by the deliberate
        rotation
        """
        if tfinal == 0.0:
            return self.chi0
        else:
            unitary_rotation = LA.expm(1j * tfinal * self.rot_basis)
            if self.noise_type == 'quasistatic':
                return unitary_rotation @ self.chi_final(tfinal) @ unitary_rotation.conj().T
            else:
                mod_interaction = unitary_rotation @ self.noise @ unitary_rotation.conj().T
                unitary_operation = LA.expm(-1j * tfinal * mod_interaction)
                return unitary_operation @ self.chi0 @ unitary_operation.conj().T

    def fidelity(self, tfinal):
        noisy_chi = self.chi_final_RF(tfinal)
        chi_product = noisy_chi @ self.chi0
        return np.trace(chi_product).real
