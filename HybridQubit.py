# Class definition for the hybrid qubit

import math
import numpy as np
import scipy.linalg as LA

import QMFormulas as qmf


def second_order_sweet_spot_match_finder(fit_params_init, operating_point, matchfreq):
    """
    Find the required parameter tunings to meet the second-order sweet spot condition.
    Inputs:
      fit_params_init: array containing:
        stsplitting (GHz)
        delta1 (GHz)
        delta2 (GHz)
      operating_point : value of detuning / stsplitting
      matchfreq : qubit frequency to match
    """

    from scipy.optimize import root

    def soss_helper(fit_params, operating_point, matchfreq):
        stsplitting, delta1, delta2 = fit_params
        h = 1e-8 * stsplitting
        ed = operating_point * stsplitting
        base_qubit = HybridQubit(ed, stsplitting, delta1, delta2)
        test_array = np.array(
            [HybridQubit(ed-2*h, stsplitting, delta1, delta2).qubit_splitting(),
             HybridQubit(ed-h, stsplitting, delta1, delta2).qubit_splitting(),
             base_qubit.qubit_splitting(),
             HybridQubit(ed+h, stsplitting, delta1, delta2).qubit_splitting(),
             HybridQubit(ed+2*h, stsplitting, delta1, delta2).qubit_splitting()]) / (2*math.pi)
        coeff_array1 = np.array([1/12, -2/3, 0, 2/3, -1/12])
        coeff_array2 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        deriv1 = np.dot(coeff_array1, test_array) / h
        deriv2 = np.dot(coeff_array2, test_array) / h**2
        # deriv1 = qmf.derivative(qubit.energy_detuning, ed, 1) / (2*math.pi)
        # deriv2 = qmf.derivative(qubit.energy_detuning, ed, 2) / (2*math.pi)
        resonance = np.abs(base_qubit.qubit_splitting()/(2*math.pi) - matchfreq)
        return [deriv1, deriv2, resonance]
    tunings = root(soss_helper,
                   fit_params_init,
                   args=(operating_point, matchfreq),
                   method='hybr', tol=1e-8,
                   options = {'eps': 1e-6}).x
    return tunings


class HybridQubit(object):
    """
    Return a hybrid qubit object
    """
    def __init__(self, ed, stsplitting, delta1, delta2):
        """
        Create an instance of a hybrid qubit
        Inputs (GHz):
          ed : dipolar detuning
          stsplitting : singlet-triplet splitting in right dot
          delta1: 0-1 tunnel coupling
          delta2: 0-2 tunnel coupling

        Also creates a dimension property that is accessible
        """
        self.ed = ed
        self.stsplitting = stsplitting
        self.delta1 = delta1
        self.delta2 = delta2
        self.dim = 3

    def hamiltonian_lab(self):
        """
        Return the hamiltonian of the hybrid qubit in units of angular GHz.
        """
        H0 = np.array([[-0.5 * self.ed, 0, self.delta1],
                       [0, -0.5*self.ed + self.stsplitting, -self.delta2],
                       [self.delta1, -self.delta2, 0.5*self.ed]])
        return 2 * math.pi * H0

    def qubit_basis(self):
        """
        Return the qubit eigenbasis
        """
        evecs = LA.eigh(self.hamiltonian_lab())[1]
        evecs = qmf.eigvector_phase_sort(evecs)
        return evecs

    def energy_detuning(self, detuning):
        """
        Return the qubit energy as a function of detuning using the
        given values of stsplitting, delta1, delta2. Primarily a convenience
        function for noise derivative calculations.
        Inputs:
          detuning [(GHz)] : dipolar detuning array
        Outputs:
          qubit energy (angular GHz)
        """
        return_array = np.zeros(len(detuning))
        for i in range(len(detuning)):
            H0 = np.array([[-0.5*detuning[i], 0, self.delta1],
                           [0, -0.5*detuning[i] + self.stsplitting, -self.delta2],
                           [self.delta1, -self.delta2, 0.5*detuning[i]]])
            evals = LA.eigvalsh(2*math.pi*H0)
            return_array[i] = evals[1] - evals[0]
        return return_array


    def energies(self):
        """
        Return an array of system energies in angular GHz
        """
        return LA.eigvalsh(self.hamiltonian_lab())

    def qubit_splitting(self):
        """
        Return the qubit splitting in angular GHz
        """
        evals = self.energies()
        return evals[1] - evals[0]


    def detuning_noise_lab(self, ded):
        """
        Return the noise matrix for detuning noise in angular GHz
        Input (GHz):
          ded: dipolar detuning
        """
        return np.diag([-0.5*ded, -0.5*ded, 0.5*ded])


    def detuning_noise_qubit(self, ded):
        """
        Return the noise matrix in the qubit frame. Will have units of 
        angular GHz.
        Input(GHz):
          ded: dipolar detuning
        """
        return self.qubit_basis().T @ self.detuning_noise_lab(ded) @ self.qubit_basis()


    def dipole_operator_qubit(self):
        """return the full dipole operator for the 
        quantum dot hybrid qubit"""
        base_operator = 0.5 * np.diag([-1, -1, 1])
        return self.qubit_basis().T @ base_operator @ self.qubit_basis()


    def splitting_derivative(self, order):
        """Calculate the nth-derivative of the qubit 
        splitting in GHz.
        Inputs:
          order: order of the derivative
        Outputs:
          d^n / d omega^n in (GHz^-(n-1))
        """
        step = 5e-3

        qm3 = HybridQubit(self.ed - 3 * step, self.stsplitting,
                          self.delta1, self.delta2)
        qm2 = HybridQubit(self.ed - 2 * step, self.stsplitting,
                          self.delta1, self.delta2)
        qm1 = HybridQubit(self.ed - 1 * step, self.stsplitting,
                          self.delta1, self.delta2)
        qp1 = HybridQubit(self.ed + 1 * step, self.stsplitting,
                          self.delta1, self.delta2)
        qp2 = HybridQubit(self.ed + 2 * step, self.stsplitting,
                          self.delta1, self.delta2)
        qp3 = HybridQubit(self.ed + 3 * step, self.stsplitting,
                          self.delta1, self.delta2)

        splitting_array = np.array([qm3.qubit_splitting(), qm2.qubit_splitting(),
                                    qm1.qubit_splitting(),
                                    self.qubit_splitting(),
                                    qp1.qubit_splitting(),
                                    qp2.qubit_splitting(), qp3.qubit_splitting()]) / (2*math.pi)
        if order == 1:
            step = 5e-3
            coeff_list = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
        elif order == 2:
            step = 5e-3
            coeff_list = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        elif order == 3:
            step = 5e-3
            coeff_list = np.array([1/8, -1, 13/8, 0, -13/8, 1, -1/8])
        else:
            pass
        
        return np.dot(splitting_array, coeff_list) / (step ** order)



class SOSSHybrid(HybridQubit):
    """Return a hybrid qubit object that is at a second order sweet spot
    resonant at some given frequency"""


    def __init__(self, ed_ratio, matchfreq):
        from scipy.interpolate import interp1d
        self.ed_ratio = ed_ratio
        self.matchfreq = matchfreq
        ed_ratio_array_ref = np.load('ed_ratio_ref.npy')
        stsplitting_array_ref = np.load('stsplitting_ref.npy')
        delta1_array_ref = np.load('delta1_ref.npy')
        delta2_array_ref = np.load('delta2_ref.npy')

        stsplitting_f = interp1d(ed_ratio_array_ref, stsplitting_array_ref)
        delta1_f = interp1d(ed_ratio_array_ref, delta1_array_ref)
        delta2_f = interp1d(ed_ratio_array_ref, delta2_array_ref)

        stsplitting = (matchfreq / 10.0) * stsplitting_f(ed_ratio)
        ed = stsplitting * ed_ratio
        delta1 = delta1_f(ed_ratio)
        delta2 = delta2_f(ed_ratio)
        super().__init__(ed, stsplitting, delta1, delta2)