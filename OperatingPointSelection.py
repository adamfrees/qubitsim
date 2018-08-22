# Script to iterate over possible hybrid qubit operating locations 
# and determine the required tolerances for the tunnel couplings

import math
import numpy as np
import scipy.linalg as LA

import HybridQubit as hybrid
import CJFidelities as cj


def two_sigma_doubling(original, sigma):
    middle = np.nonzero(np.fabs(original) <= 2*sigma)[0]
    new_size = len(original) + len(middle) - 1
    start = range(0, middle[0])
    finish = range(middle[-1]+1, len(original))
    new_start = start
    new_middle = range(middle[0], middle[0]+2*len(middle)-1)
    new_finish = range(new_middle[-1]+1, new_size)
    new_samples = noise_doubling(original[middle])[1]

    new_array = np.zeros((new_size))
    new_array[new_start] += original[start]
    new_array[new_middle] += new_samples
    new_array[new_finish] += original[finish]
    return new_array


def wing_doubling(original, sigma):
    middle = np.nonzero(np.fabs(original) <= 2*sigma)[0]
    start = range(0, middle[0])
    finish = range(middle[-1]+1, len(original))
    new_size = len(original) + len(start) + len(finish) - 2

    start_double = noise_doubling(original[start])[1]
    finish_double = noise_doubling(original[finish])[1]

    new_array = np.zeros((new_size))
    new_array[0:len(start_double)] += start_double
    new_array[len(start_double):len(start_double)+len(middle)] += original[middle]
    new_array[len(start_double)+len(middle):] += finish_double
    return new_array


def noise_sample(qubit, tfinal, ded):
    H0 = np.diag(qubit.energies())
    noise = qubit.detuning_noise_qubit(ded)
    indices = [0, 1]

    ChoiSimulation = cj.CJ(indices, H0, noise)
    if tfinal == 0:
        return ChoiSimulation.chi0
    else:
        return ChoiSimulation.chi_final_RF(tfinal)

def noise_iteration(qubit, tfinal, noise_samples):
    cj_array = np.zeros((9, 9, len(noise_samples), dtype=complex)
    for i in range(len(noise_samples)):
        ded = noise_samples[i]
        cj_array[:, :, i] += noise_sample(qubit, tfinal, ded)
    return cj_array


def noise_averaging(x, noise_weights, cj_array):
    from scipy.integrate import simps
    norm = simps(noise_weights, x)
    matrix_int = simps(np.multiply(cj_array, noise_weights), x)
    return matrix_int / norm


def multi_sigma_noise_sampling(qubit, tfinal, sigma_array):
    from scipy.stats import norm
    noise_samples0 = np.linspace(-5*sigma_array[-1], 5*sigma_array[-1], 101)
    noise_weights = np.zeros((len(sigma_array), len(noise_samples0)))
    for i in range(len(sigma_array)):
        noise_weights[i, :] += norm.pdf(noise_samples0, loc=0.0, scale=sigma_array[i])
    
    converge_value = 1.0
    num_runs = 1

    cj_array0 = noise_iteration(qubit, tfinal, noise_samples0)

def multi_sigma_convergence(tfinal, sigma_array, qubit):
    """Ensure convergence of averaging a computed chi-matrix array with 
    respect to gaussians with standard deviations given by sigma_array.
    If convergence hasn't been reached, more samples will be taken."""
    initial_samples = np.linspace(-5*sigma_array[-1], 5*sigma_array[-1], 101)
    cj_sample_array = np.zeros((9, 9, len(initial_samples), dtype=complex)
    return None


def time_sweep(qubit):
    """
    Given a qubit operating point, represented by the input hybrid qubit object,
    return the process matrix evolution as a function of time.
    
    Inputs: qubit
    
    Ouputs: trange, array of simulated times
            chi_array, array of process matrices at the simulated times
    """
    return None

def operating_point_stability(operating_point, match_freq):
    """
    Given an operating point in the detuning, want to find the required 
    tolerance in the tunnel couplings. The metric will be the decrease in 
    the coherence time. We will use variations of plus/minus 
    1, 2, 3, 4, 5 percent in each tunnel coupling. For each operating point, 
    we will take the 0 percent variation as the reference coherence time. 
    Then, we will extract the coherence time matrix for 0..5 percent in delta1 
    by pm 0..5 percent in delta2

    Inputs:
      operating_point: value of epsilon/EST
      match_freq: assuming resonant operation, the frequency of the CPW

    Outputs:
      delta1_array: tested values of delta1
      delta2_array: tested values of delta2
      coherence_array: array of coherence times with shape 
                       (len(delta1_array), len(delta2_array))
    """

    ref_hybrid = hybrid.SOSSHybrid(operating_point, match_freq)
    delta1_array = np.arange(0.95, 1.05, 0.01) * ref_hybrid.delta1
    delta2_array = np.arange(0.95, 1.05, 0.01) * ref_hybrid.delta2

    coherence_array = np.zeros((len(delta1_array), len(delta2_array)))
    for i in range(len(delta1_array)):
        for j in range(len(delta2_array)):
            local_hybrid = hybrid.HybridQubit(ref_hybrid.ed,
                                              ref_hybrid.stsplitting,
                                              delta1_array[i],
                                              delta2_array[j])
            
    return None