# Class definition for the charge dipole qubit

import math
import numpy as np
import scipy.linalg as LA


class ChargeDipole(object):
    """
    Return a charge dipole qubit object
    """
    def __init__(self, ed, delta):
        """
        Create an instance of a hybrid qubit. The accessible features 
        are ed, stsplitting, delta1, delta2, and dim.

        Parameters
        ----------
        ed : float
            dipolar detuning
            Units: GHz
        delta : float
            tunnel coupling
            Units: GHz
        
        Returns
        -------
        self : Hybrid qubit object
        """
        __slots__ = 'ed', 'delta',
        self.ed = ed
        self.delta = delta
        self.dim = 2

    def hamiltonian_lab(self):
        """
        Generate the hamiltonian of the hybrid qubit in the laboratory or 
        charge basis. 

        Parameters
        ----------
        self : charge dipole qubit object

        Returns
        -------
        H0 : (3, 3) float
            Hybrid qubit hamiltonian in the lab frame
            Units: rad/ns
        """
        H0 = np.array([[-0.5 * self.ed, self.delta],
                       [self.delta, 0.5*self.ed]])
        return 2 * math.pi * H0


    def c_detune_qubit_splitting(self, deviation):
        """
        From the base hamiltonian, find the complex form of the qubit's 
        energy resulting from a complex deviation in the detuning.

        Parameters
        ----------
        deviation: complex
            a complex deviation in the dipolar detuning
            Units: GHz
        
        Returns
        -------
        complex
            qubit splitting energy 
            units: GHz
        """
        Hbase = self.hamiltonian_lab() / (2*math.pi)
        Hdeviation = deviation * np.diag([-0.5, 0.5])
        evals = np.sort(LA.eigvals(Hbase + Hdeviation))
        return evals[1] - evals[0]


    def qubit_basis(self):
        """
        Return the qubit eigenbasis sorted according to the convention 
        that the first element of every eigenvector must be positive.

        Returns
        -------
        (3, 3) float array
            Matrix whose columns are normalized eigenvectors of the system
        """
        evecs = LA.eigh(self.hamiltonian_lab())[1]
        evecs = ChargeDipole.eigvector_phase_sort(evecs)
        return evecs


    def energies(self):
        """
        Calculate the system energies from the given Hamiltonian.

        Returns
        -------
        (3,) float array
            array of system energies
            Units: rad/ns
        """
        return LA.eigvalsh(self.hamiltonian_lab())


    def qubit_splitting(self):
        """
        Return the qubit splitting in rad/ns

        Returns
        -------
        float
            qubit energy splitting
            Units: rad/ns
        """
        evals = self.energies()
        return evals[1] - evals[0]


    def detuning_noise_lab(self, ded):
        """
        Return the noise matrix for detuning noise in rad/ns

        Parameters
        ----------
        ded : float
            diploar detuning
            Units: GHz
        
        Returns
        -------
        (3, 3) float array
            lab frame detuning perturbation
            Units: rad/ns
        """
        return 2*math.pi*np.diag([-0.5*ded, 0.5*ded])


    def detuning_noise_qubit(self, ded):
        """
        Return the noise matrix in the qubit frame.

        Parameters
        ----------
        ded : float
            dipolar detuning 
            Units: GHz
        Returns
        -------
        (3,3) float array
            detuning noise matrix in the qubit basis.
            Units: rad/ns
        """
        return self.qubit_basis().T @ self.detuning_noise_lab(ded) @ self.qubit_basis()


    def dipole_operator_qubit(self):
        """
        Calculate the full dipole operator for the hybrid qubit in 
        the qubit basis
        
        Returns
        -------
        (3, 3) float array
            dipole operator
            Units: dimensionless
        """
        base_operator = 0.5 * np.diag([-1, 1])
        return self.qubit_basis().T @ base_operator @ self.qubit_basis()


    def splitting_derivative(self, order):
        """
        Calculate the nth-derivative of the qubit splitting.
        
        The derivative methods below rely on complex step derivatives for 
        numerical stability.

        Parameters
        ----------
        order : int
            order of the derivative (1st, 2nd, 3rd)
        
        Returns
        -------
        derivative : float
            Units: GHz^(order - 1)
        """
        
        if order == 1:
            step = 1e-10
            c_num = step * (1 + 1.j) / math.sqrt(2)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(-c_num)
            return np.imag(eval1 - eval2) / (step * math.sqrt(2))
        elif order == 2:
            step = 1e-8
            c_num = step * (1 + 1.j) / math.sqrt(2)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(-c_num)
            return np.imag(eval1 + eval2) / step**2
        elif order == 3:
            step = 1e-8
            c_num = step * (1 + 1.j)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(0.5*c_num)
            eval3 = self.c_detune_qubit_splitting(-0.5*c_num)
            eval4 = self.c_detune_qubit_splitting(c_num)
            return math.sqrt(2)*np.imag(eval1 - 2*eval2 + 2*eval3 - eval4)
        else:
            return None


    @staticmethod
    def eigvector_phase_sort(eig_matrix):
        for i in range(eig_matrix.shape[1]):
            if eig_matrix[0, i] < 0:
                eig_matrix[:, i] *= -1
        return eig_matrix