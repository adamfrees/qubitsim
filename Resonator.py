# Source code for a resonator class

import numpy as np

class Resonator(object):
    """Return a microwave frequency resonator object"""
    def __init__(self, omega, dim):
        self.omega = omega
        self.dim = dim

    def number_operator(self):
        """Return the number operator for the resonator"""
        return np.diag(np.arange(0, self.dim, 1))

    def annihilation_operator(self):
        """Return the annihilation operator for the resonator"""
        return np.diag(np.sqrt(np.arange(1, self.dim)), k=1)

    def creation_operator(self):
        """Return the creation operator for the resonator"""
        return np.diag(np.sqrt(np.arange(1, self.dim, 1)), k=-1)