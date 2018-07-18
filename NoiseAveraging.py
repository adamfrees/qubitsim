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
    operating_point = 3.0
    match_freq = 10.0
    indices = [0, 1]
    qubit = hybrid.SOSSHybrid(operating_point, match_freq)
    H0 = qubit.hamiltonian_lab()
    noise = qubit.detuning_noise_lab(ded)

    ChoiSimulation = cj.CJ(indices, H0+noise)
    if tfinal == 0:
        return ChoiSimulation.chi0
    else:
        return ChoiSimulation.chi_final(tfinal)


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
    diff_array = np.diff(x)
    if np.allclose(np.diff(diff_array), np.zeros((len(x) - 2))):
        # In this case, equal spacing was used for x
        # allowing a faster trapz implementation
        dx = np.mean(diff_array)
        norm = np.trapz(noise_samples, dx=dx)
        matrix_int = np.trapz(np.multiply(cj_array, noise_samples), dx=dx)
        return matrix_int / norm
    else:
        # We got less lucky, so we'll use the more general method
        norm = np.trapz(noise_samples, x=x)
        matrix_int = np.trapz(np.multiply(cj_array, noise_samples), x=x)
        return matrix_int / norm


def simple_noise_sampling(tfinal, samples0=15):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    ueV_conversion = 0.241799050402417
    sigma = 5.0 * ueV_conversion
    x0 = np.linspace(-7 * sigma, 7*sigma, samples0)
    noise_samples0 = qmf.gaussian(x0, 0.0, sigma)
    cj_array0 = noise_iteration(noise_samples0, tfinal)
    average_cj0 = noise_averaging(x0, noise_samples0, cj_array0)
    converge_value = 1.0
    while converge_value > 1e-8:
        x1new, x1full = noise_doubling(x0)
        noise_samples1new = qmf.gaussian(x1new, 0.0, sigma)
        noise_samples1full = qmf.gaussian(x1full, 0.0, sigma)
        cj_array1new = noise_iteration(noise_samples1new, tfinal)
        cj_array1 = np.zeros((9, 9, len(x1full)), dtype=complex)
        cj_array1[:, :, ::2] = cj_array0
        cj_array1[:, :, 1::2] = cj_array1new
        average_cj1 = noise_averaging(x1full, noise_samples1full, cj_array1)
        converge_value = qmf.processInfidelity(average_cj0, average_cj1)

        print(converge_value)

        samples = len(noise_samples0)
        print(samples)

        x0 = x1full
        noise_samples0 = noise_samples1full
        cj_array0 = cj_array1
        average_cj0 = average_cj1
    return average_cj1, samples


def bare_time_evolution():
    trange = np.linspace(0, 500, 30)
    cj_time_array = np.zeros((9, 9, 20), dtype=complex)
    samples = 15
    for i in range(20):
        cj_average, samples = simple_noise_sampling(trange[i], samples)
        print(samples)
        cj_time_array[:, :, i] += cj_average
    return trange, cj_time_array


if __name__ == '__main__':
    trange, cj_time_array = bare_time_evolution()
    np.save('trange_test.npy', trange)
    np.save('cj_array_test.npy', cj_time_array)