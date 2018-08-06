# Calculating Noise-averaged time evolution 
# using Choi-Jamiolkowski isomorphisms

import math
import numpy as np

import QMFormulas as qmf
import HybridQubit as hybrid
import CJFidelities as cj


def noise_doubling(original):
    """Bisect an original sample given"""
    new_samples = 0.5 * np.diff(original) + original[:-1]
    full_array = np.zeros((original.size + new_samples.size,), dtype=original.dtype)
    full_array[0::2] = original
    full_array[1::2] = new_samples
    return new_samples, full_array


def noise_sample_run(ded, tfinal):
    """Run a single noise sample at noise point ded (GHz)
    and at time tfinal"""
    operating_point = 5.0
    match_freq = 10.0
    indices = [0, 1]
    qubit = hybrid.SOSSHybrid(operating_point, match_freq)
    H0 = qubit.hamiltonian_lab()
    noise = qubit.detuning_noise_lab(ded)

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


def simple_noise_sampling(tfinal, samples0=15):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    from scipy.linalg import sqrtm
    ueV_conversion = 0.241799050402417
    sigma = 5.0 * ueV_conversion
    x0 = np.linspace(-10 * sigma, 10*sigma, samples0)
    noise_samples0 = qmf.gaussian(x0, 0.0, sigma)
    cj_array0 = noise_iteration(noise_samples0, tfinal)
    average_cj0 = noise_averaging(x0, noise_samples0, cj_array0)
    converge_value = 1.0

    while converge_value > 1e-12:
        x1new, x1full = noise_doubling(x0)
        noise_samples1new = qmf.gaussian(x1new, 0.0, sigma)
        noise_samples1full = qmf.gaussian(x1full, 0.0, sigma)
        cj_array1new = noise_iteration(noise_samples1new, tfinal)
        cj_array1 = np.zeros((9, 9, len(x1full)), dtype=complex)
        cj_array1[:, :, ::2] = cj_array0
        cj_array1[:, :, 1::2] = cj_array1new
        average_cj1 = noise_averaging(x1full, noise_samples1full, cj_array1)
        print(qmf.processInfidelity(average_cj0, average_cj1))
        converge_value = np.abs(np.trace(sqrtm((average_cj0-average_cj1) @ (average_cj0.T - average_cj1.T))))

        samples = len(noise_samples0)

        x0 = x1full
        noise_samples0 = noise_samples1full
        cj_array0 = cj_array1
        average_cj0 = average_cj1
    return average_cj1, samples


def bare_time_evolution():
    tsteps = 100
    trange = np.linspace(0, 30, tsteps)
    cj_time_array = np.zeros((9, 9, tsteps), dtype=complex)
    samples = 81
    for i in range(tsteps):
        print('Time step: {}'.format(i))
        cj_average, samples = simple_noise_sampling(trange[i], samples)
        cj_time_array[:, :, i] += cj_average
    return trange, cj_time_array


def choosing_final_time(qubit, sigma):
    """ Function to make a guess at the final time required 
    for estimating decoherence"""
    planck = 4.135667662e−9
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

    T21 = (deriv1*sigma) / (math.sqrt(2) * h)
    T22 = (deriv2 * sigma**2) / (math.sqrt(2) * h**2)
    T23 = (deriv3 * sigma**3) / (math.sqrt(2) * h**3)
    return np.array([T21, T22, T23])


if __name__ == '__main__':
    trange, cj_time_array = bare_time_evolution()
    np.save('trange_test.npy', trange)
    np.save('cj_array_test.npy', cj_time_array)