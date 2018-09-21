# Source code for the charge quadrupole qubit class

import math
import numpy as np
import scipy.linalg as LA

class CQ(object):
    """
    Return a charge quadrupole qubit object
    """
    def __init__(self, eq, ed, delta1, delta2):
        """
        Initilizes the qubit
        Inputs:
          eq: quadrupolar detuning in GHz
          delta: tunnel coupling in GHz
        """
        self.eq = eq
        self.ed = ed
        self.delta1 = delta1
        self.delta2 = delta2
        self.dim = 3
    

    def hamiltonian_lab(self):
        """
        Returns the hamiltonian in the labframe in units of rad/ns
        """
        H0 = np.array([[0.5 * self.eq, (self.delta1 + self.delta2)/math.sqrt(2), (self.delta1 - self.delta2)/math.sqrt(2)],
                      [(self.delta1 + self.delta2)/math.sqrt(2), -0.5 * self.eq, self.ed],
                      [(self.delta1 - self.delta2)/math.sqrt(2), self.ed, 0.5 * self.eq]])
        return 2 * math.pi * H0


    def dim_sort(self):
        """
        The cq qubit has an odd energy structure, this gives 
        the indices to sort eigenvectors and eigenvalues into 
        qubit + leakage ordering
        """
        return [0, 2, 1]


    def energies(self):
        """
        Return an array of system energies in rad/ns
        """
        return LA.eigvalsh(self.hamiltonian_lab())[self.dim_sort()]


    def qubit_splitting(self):
        """
        Return the qubit splitting in rad/ns
        """
        evals = self.energies()
        return evals[1] - evals[0]


    def detuning_noise_lab(self, ded):
        """
        Return the noise matrix for detuning noise in rad/ns
        Input:
          ded: dipolar detuning noise in GHz
        Output:
          (3x3) array in rad/ns
        """
        noise = np.array([[0, 0, 0], [0, 0, ded], [0, ded, 0]])
        return 2*math.pi*noise



class IdealCQ(CQ):
    """
    Return a cq object with ed = 0 and balanced tunnel couplings
    """
    def __init__(self, eq, delta):
        self.eq = eq
        self.delta = delta
        super().__init__(eq, 0.0, delta, delta)