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
        Returns the hamiltonian in the labframe in units of radians/ns
        """
        H0 = np.array([[0.5 * self.eq, (self.delta1 + self.delta2)/math.sqrt(2), (self.delta1 - self.delta2)/math.sqrt(2)],
                      [(self.delta1 + self.delta2)/math.sqrt(2), -0.5 * self.eq, self.ed],
                      [(self.delta1 - self.delta2)/math.sqrt(2), self.ed, 0.5 * self.eq]])
        return 2 * math.pi * H0