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
        """
        self.ed = ed
        self.stsplitting = stsplitting
        self.delta1 = delta1
        self.delta2 = delta2

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
            H0 = 2*math.pi*np.array([[-0.5*detuning[i], 0, self.delta1],
                                     [0, -0.5*detuning[i] + self.stsplitting, -self.delta2],
                                     [self.delta1, -self.delta2, 0.5*detuning[i]]])
            evals = LA.eigvalsh(H0)
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


class SOSSHybrid(HybridQubit):
    """Return a hybrid qubit object that is at a second order sweet spot
    resonant at some given frequency"""


    def __init__(self, ed_ratio, matchfreq, guess_array=[10.0, 6.4, 6.6]):
        self.ed_ratio = ed_ratio
        self.matchfreq = matchfreq
        ed_testing_ratio = np.arange(12.0, ed_ratio+0.01, -0.01)
        for ed in ed_testing_ratio:
            tunings = second_order_sweet_spot_match_finder(guess_array, ed, matchfreq)
            guess_array = tunings
        stsplitting, delta1, delta2 = tunings
        ed = ed_ratio * stsplitting
        super().__init__(ed, stsplitting, delta1, delta2)