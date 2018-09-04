# Script to iterate over possible hybrid qubit operating locations 
# and determine the required tolerances for the tunnel couplings

import math
import numpy as np
import scipy.linalg as LA

import HybridQubit as hybrid
import CJFidelities as cj


def choosing_final_time(qubit, sigma):
    """ Function to make a guess at the final time required 
    for estimating decoherence"""
    h = 1e-2
    coeff_array1 = np.array([1/12, -2/3, 0, 2/3, -1/12])
    coeff_array2 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
    coeff_array3 = np.array([-1/2, 1, 0, -1, 1/2])

    ed = qubit.ed
    stsplitting = qubit.stsplitting
    delta1 = qubit.delta1
    delta2 = qubit.delta2

    qsm2 = hybrid.HybridQubit(ed - 2*h, stsplitting, delta1, delta2).qubit_splitting()
    qsm1 = hybrid.HybridQubit(ed - h, stsplitting, delta1, delta2).qubit_splitting()
    qsp1 = hybrid.HybridQubit(ed + h, stsplitting, delta1, delta2).qubit_splitting()
    qsp2 = hybrid.HybridQubit(ed + 2*h, stsplitting, delta1, delta2).qubit_splitting()

    sample_array = np.array([qsm2, qsm1, qubit.qubit_splitting(), qsp1, qsp2]) / (2*math.pi)

    deriv1 = np.abs(np.dot(sample_array, coeff_array1))
    deriv2 = np.abs(np.dot(sample_array, coeff_array2))
    deriv3 = np.abs(np.dot(sample_array, coeff_array3))

    G21 = 2*(deriv1*sigma)
    G22 = 2*(deriv2 * sigma**2)
    G23 = 2*(deriv3 * sigma**3)
    return np.sum(np.reciprocal(np.array([G21, G22, G23])))


def noise_doubling(original):
    """Bisect an original sample given"""
    new_samples = 0.5 * np.diff(original) + original[:-1]
    full_array = np.zeros((original.size + new_samples.size,), dtype=original.dtype)
    full_array[0::2] = original
    full_array[1::2] = new_samples
    return new_samples, full_array


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
    cj_array = np.zeros((9, 9, len(noise_samples)), dtype=complex)
    for i in range(len(noise_samples)):
        ded = noise_samples[i]
        cj_array[:, :, i] += noise_sample(qubit, tfinal, ded)
    return cj_array


def noise_averaging(x, noise_weights, cj_array):
    from scipy.integrate import simps
    norm = simps(noise_weights, x)
    matrix_int = simps(np.multiply(cj_array, noise_weights), x)
    return matrix_int / norm


def process_noise(qubit, tstep, noise_samples, sigma_array):
    """
    Input the qubit reference, the time for the evolution, the noise samples considered,
    and the standard deviations of the noise.

    Returns two arrays:
      average_chi_array: (len(sigma_array), qubit_dim, qubit_dim) sized array 
                         containing the chi-matrix at this time for each sigma
      raw_chi_array: (len(noise_samples), qubit_dim, qubit_dim) sized array
                     raw samples used for the averaging, required to 
                     save time in recomputation
    """
    from scipy.stats import norm
    noise_weights = np.zeros((len(sigma_array), len(noise_samples)))
    average_chi_array = np.zeros((len(sigma_array), 9,9), dtype=complex)
    raw_chi_array = noise_iteration(qubit, tstep, noise_samples)
    for i in range(len(sigma_array)):
        noise_weights[i, :] += norm.pdf(noise_samples, loc=0.0, scale=sigma_array[i])
        average_chi_array[i, :, :] += noise_averaging(noise_samples, noise_weights[i, :], raw_chi_array)
    return average_chi_array, raw_chi_array
    


def multi_sigma_noise_sampling(qubit, tstep, sigma_array, num_samples):
    """Ensure convergence of averaging a computed chi-matrix array with 
    respect to gaussians with standard deviations given by sigma_array.
    If convergence hasn't been reached, more samples will be taken."""

    noise_samples0 = np.linspace(-5*sigma_array[-1], 5*sigma_array[-1], num_samples)
    average_chi_array0, raw_chi_array0 = process_noise(qubit, tstep, noise_samples0, sigma_array)
    
    converge_value = 1.0
    num_runs = 1
    # Used to progressively refine the sampling space
    sig_index = -1

    while converge_value > 1e-7:
        if num_runs % 3 == 0:
            noise_samples1 = wing_doubling(noise_samples0, sigma_array[sig_index])
        else:
            noise_samples1 = two_sigma_doubling(noise_samples0, sigma_array[sig_index])
        average_chi_array1, raw_chi_array1 = process_noise(qubit, tstep, noise_samples1, sigma_array)

        converge_array = np.zeros((len(sigma_array)))

        diff_matrix = average_chi_array1 - average_chi_array0
        converge_array = np.sqrt(
            np.einsum('ijj',
            np.einsum('ijk,ikm->ijm', diff_matrix, 
                np.einsum('ikj', diff_matrix.conj()))))
  
        # Ensure that all of the individual chi-matrices have converged
        converge_value = np.max(converge_array)
        for i, norm in reversed(list(enumerate(converge_array))):
            if norm < 1e-8:
                sig_index = i
                break

        noise_samples0 = noise_samples1
        average_chi_array0 = average_chi_array1
        raw_chi_array0 = raw_chi_array1
        num_runs += 1
    return len(noise_samples1), average_chi_array1


def time_sweep(qubit):
    """
    Given a qubit operating point, represented by the input hybrid qubit object,
    return the process matrix evolution as a function of time.
    
    Inputs: qubit
    
    Ouputs: trange, array of simulated times
            chi_array, array of process matrices at the simulated times
    """
    ueV_conversion = 0.241799050402417
    sigma_array = np.array([0.5, 1.0, 2.0, 5.0, 7.0, 10.0]) * ueV_conversion
    tfinal = choosing_final_time(qubit, np.median(sigma_array))
    tmin = choosing_final_time(qubit, np.max(sigma_array))
    tarray = np.arange(0.0, tfinal, tmin / 100)
    # tarray = np.linspace(0.0, tfinal, 1000)
    num_noise_samples = 32

    mass_chi_array = np.empty((len(tarray), len(sigma_array), 9, 9), dtype=complex)
    for i in range(len(tarray)):
        tstep = tarray[i]
        print(tstep)
        num_noise_samples, sigma_chi_array = multi_sigma_noise_sampling(qubit, tstep, sigma_array, num_noise_samples)
        mass_chi_array[i, :, :, :] = sigma_chi_array

    return tarray, sigma_array, mass_chi_array


def time_evolution_point(operating_point, delta1_point, delta2_point):
    """At the given operating point, return the time array and chi_matrix 
    array for each of the noise values considered"""
    match_freq = 10.0
    qubit_base = hybrid.SOSSHybrid(operating_point, match_freq)
    delta1_ref = qubit_base.delta1
    delta2_ref = qubit_base.delta2
    stsplitting_ref = qubit_base.stsplitting
    ed_ref = qubit_base.ed
    qubit = hybrid.HybridQubit(ed_ref,
                               delta1_point * delta1_ref,
                               delta2_point * delta2_ref,
                               stsplitting_ref)
    trange, sigma_array, mass_chi_array = time_sweep(qubit)
    return trange, sigma_array, mass_chi_array