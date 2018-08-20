# Calculating Noise-averaged time evolution 
# using Choi-Jamiolkowski isomorphisms

import math
import numpy as np

import QMFormulas as qmf
import HybridQubit as hybrid
import CJFidelities as cj


def even_area_sampling(samples, sigma):
    """Use the percentile point function of the normal distribution 
    to return the sample points that represent equal area subdivisions
    underneath a gaussian distribution with mean=0 and sigma=sigma"""
    from scipy.stats import norm
    samples = norm.ppf(np.linspace(0, 1, samples), 0.0, sigma)
    return samples[1:-1]

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



def noise_sample_run(ded, tfinal):
    """Run a single noise sample at noise point ded (GHz)
    and at time tfinal"""
    operating_point = 3.0
    match_freq = 10.0
    indices = [0, 1]
    qubit = hybrid.SOSSHybrid(operating_point, match_freq)
    H0 = np.diag(qubit.energies())
    noise = qubit.detuning_noise_qubit(ded)
    # noise = np.zeros((3,3))

    ChoiSimulation = cj.CJ(indices, H0, noise)
    if tfinal == 0:
        return ChoiSimulation.chi0
    else:
        return ChoiSimulation.chi_final_RF(tfinal)


def noise_iteration(noise_samples, tfinal):
    """
    Given an array of noise values to sample and a simulation time,
    return the array of process matrices corresponding to this solution.
    """
    cj_array = np.zeros((9, 9, noise_samples.shape[0]), dtype=complex)
    for i in range(noise_samples.shape[0]):
        ded = noise_samples[i]
        cj_array[:, :, i] += noise_sample_run(ded, tfinal)
    return cj_array


def noise_averaging(x, noise_samples, cj_array):
    from scipy.integrate import simps
    # norm = np.trapz(noise_samples, x=x)
    norm = simps(noise_samples, x)
    # matrix_int = np.trapz(np.multiply(cj_array, noise_samples), x=x)
    matrix_int = simps(np.multiply(cj_array, noise_samples), x)
    return matrix_int / norm


def simple_noise_sampling(tfinal, samples0):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    from scipy.linalg import sqrtm
    ueV_conversion = 0.241799050402417
    sigma = 5.0 * ueV_conversion
    noise_samples0 = np.linspace(-5 * sigma, 5*sigma, samples0)
    noise_weights0 = qmf.gaussian(noise_samples0, 0.0, sigma)
    cj_array0 = noise_iteration(noise_samples0, tfinal)
    average_cj0 = noise_averaging(noise_samples0, noise_weights0, cj_array0)
    converge_value = 1.0

    while converge_value > 1e-9:
        noise_samples1_new, noise_samples1 = noise_doubling(noise_samples0)
        noise_weights1 = qmf.gaussian(noise_samples1, 0.0, sigma)
        cj_array1_new = noise_iteration(noise_samples1_new, tfinal)
        cj_array1 = np.zeros((9, 9, len(noise_samples1)), dtype=complex)
        cj_array1[:, :, ::2] += cj_array0
        cj_array1[:, :, 1::2] += cj_array1_new
        average_cj1 = noise_averaging(noise_samples1, noise_weights1, cj_array1)
        print(qmf.processInfidelity(average_cj0, average_cj1))
        converge_value = np.abs(np.trace(sqrtm((average_cj0-average_cj1) @ (average_cj0.T - average_cj1.T))))
        print(converge_value)
        samples = len(noise_samples0)
        print(samples)

        noise_samples0 = np.copy(noise_samples1)
        cj_array0 = np.copy(cj_array1)
        average_cj0 = np.copy(average_cj1)

    return average_cj1, samples


def even_area_noise_sampling(tfinal, samples0):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    from scipy.linalg import sqrtm
    print(samples0)
    ueV_conversion = 0.241799050402417
    sigma = 5.0 * ueV_conversion
    noise_samples0 = even_area_sampling(samples0, sigma)
    noise_weights0 = qmf.gaussian(noise_samples0, 0.0, sigma)
    cj_array0 = noise_iteration(noise_samples0, tfinal)
    average_cj0 = noise_averaging(noise_samples0, noise_weights0, cj_array0)
    converge_value = 1.0

    while converge_value > 1e-10:
        noise_samples1 = even_area_sampling(samples0+40, sigma)
        noise_weights1 = qmf.gaussian(noise_samples1, 0.0, sigma)
        cj_array1 = noise_iteration(noise_samples1, tfinal)
        average_cj1 = noise_averaging(noise_samples1, noise_weights1, cj_array1)
        print(samples0)
        # print(qmf.processInfidelity(static_start_cj0, average_cj1))
        converge_value = np.abs(np.trace(sqrtm((average_cj0-average_cj1) @ (average_cj0.T - average_cj1.T))))
        print(converge_value)
        if converge_value > 1e-10:
            samples0 += 40
        average_cj0 = average_cj1
    return average_cj1, samples0


def bare_time_evolution():
    tsteps = 100
    trange = np.linspace(0, 300, tsteps)
    cj_time_array = np.zeros((9, 9, tsteps), dtype=complex)
    samples = 21
    for i in range(tsteps):
        print('Time step: {}'.format(i))
        cj_average, samples = simple_noise_sampling(trange[i], samples)
        cj_time_array[:, :, i] += cj_average
    return trange, cj_time_array


def choosing_final_time(qubit, sigma):
    """ Function to make a guess at the final time required 
    for estimating decoherence"""
    planck = 4.135667662e-9
    h = 1e-3
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

    T21 = (deriv1*sigma) / (math.sqrt(2) * planck)
    T22 = (deriv2 * sigma**2) / (math.sqrt(2) * planck**2)
    T23 = (deriv3 * sigma**3) / (math.sqrt(2) * planck**3)
    return np.sum(np.array([T21, T22, T23]))