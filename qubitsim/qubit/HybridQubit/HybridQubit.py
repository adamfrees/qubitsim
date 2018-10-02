# Class definition for the hybrid qubit

import math
import numpy as np
import scipy.linalg as LA

def eigvector_phase_sort(eig_matrix):
    for i in range(eig_matrix.shape[1]):
        if eig_matrix[0, i] < 0:
            eig_matrix[:, i] *= -1
    return eig_matrix

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


    def c_detune_qubit_splitting(self, deviation):
        """
        From the base hamiltonian, find the complex form of the qubit's 
        energy resulting from a complex deviation in the detuning.

        Inputs:
          deviation: a complex quantitity
        Outputs:
          Complex energy with each component in GHz
        """
        Hbase = self.hamiltonian_lab() / (2*math.pi)
        Hdeviation = deviation * np.diag([-0.5, -0.5, 0.5])
        evals = np.sort(LA.eigvals(Hbase + Hdeviation))
        return evals[1] - evals[0]


    def qubit_basis(self):
        """
        Return the qubit eigenbasis
        """
        evecs = LA.eigh(self.hamiltonian_lab())[1]
        evecs = eigvector_phase_sort(evecs)
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
        Output:
          (3x3) array in rad/ns representing lab frame detuning perturbation
        """
        return 2*math.pi*np.diag([-0.5*ded, -0.5*ded, 0.5*ded])


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
        splitting in GHz. The derivative methods 
        below rely on complex step derivatives for greater 
        accuracy.

        Inputs:
          order: order of the derivative
        Outputs:
          d^n / d omega^n in (GHz^-(n-1))
        """
        
        if order == 1:
            step = 1e-7
            # coeff_list = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
            return np.imag(self.c_detune_qubit_splitting(step)) / step
        elif order == 2:
            step = 1e-6
            c_num = step * (1 + 1.j) / math.sqrt(2)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(-c_num)
            return np.imag(eval1 + eval2) / step**2
        elif order == 3:
            step = 1e-5
            c_num = step * (1 + 1.j)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(0.5*c_num)
            eval3 = self.c_detune_qubit_splitting(-0.5*c_num)
            eval4 = self.c_detune_qubit_splitting(c_num)
            return math.sqrt(2)*np.imag(eval1 - 2*eval2 + 2*eval3 - eval4)
        else:
            return None


class SOSSHybrid(HybridQubit):
    """Return a hybrid qubit object that is at a second order sweet spot
    resonant at some given frequency"""


    def __init__(self, ed_ratio, matchfreq):
        """
        Create a SOSS hybrid qubit object. It inherits all hybrid qubit
        class methods and has the additional specifications of a particular
        operating point in detuning space and the frequency must match the 
        given value.

        Inputs:
          ed_ratio: ed / stsplitting, the operating point in detuning space
          matchfreq: the specified qubit frequency
        """ 
        self.ed_ratio = ed_ratio
        self.matchfreq = matchfreq

        guess = [0.7 * matchfreq, 0.7 * matchfreq, matchfreq]
        delta1, delta2, stsplitting = SOSSHybrid.__find_sweet_spot(guess, ed_ratio, matchfreq)
        ed = ed_ratio.stsplitting
        super().__init__(ed, stsplitting, delta1, delta2)


    @staticmethod
    def __conditions(vector_x, operating_ratio, res_freq):
        """
        This function calculates the required values that 
        need to be zero for the SOSS qubit

        Inputs:
          vector_x: [delta1, delta2, stsplitting]
          operating_ratio: ed / stsplitting used for operation
          res_freq: qubit frequency to match
        Output:
          [resonance, D1, D2]
          resonance: need to match the given frequency
          D1: first derivative of qubit spectrum
          D2: second derivative of qubit spectrum
        """
        delta1, delta2, stsplitting = vector_x
        ref_qubit = super().__init__(operating_ratio * stsplitting,
                                     stsplitting,
                                     delta1,
                                     delta2)
        resonance = res_freq - ref_qubit.qubit_splitting() / (2*math.pi)
        D1 = ref_qubit.splitting_derivative(1)
        D2 = ref_qubit.splitting_derivative(2)
        return [resonance, D1, D2]


    @staticmethod
    def __find_sweet_spot(guess, operating_ratio, res_freq):
        """
        This method uses the hybrid method and 
        scipy.optimize.root to find the values of 
        delta1, delta2, and stsplitting that result in a 
        second order sweet spot (SOSS).

        Inputs:
          guess: [delta1, delta2, stsplitting]
            a guess for the starting values
          operating_ratio:  ed / stsplitting for operation
          res_freq: the required qubit frequency
        Ouputs:
          tunings: [delta1, delta2, stsplitting]
          The values found by root to be a SOSS
        """
        from scipy.optimize import root
        tunings = root(SOSSHybrid.__conditions,
                   guess,
                   args=(operating_ratio, res_freq),
                   method='hybr', tol=1e-8,
                   options = {'eps': 1e-6}).x
        return tunings