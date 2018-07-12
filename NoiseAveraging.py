# Calculating Noise-averaged time evolution 
# using Choi-Jamiolkowski isomorphisms

import math
import numpy as np

import QMFormulas as qmf
import HybridQubit as hybrid
import CJFidelities as cj


def noise_doubling(original):
    """Bisect an original sample given"""
    new_samples = 0.5 * np.diff(original) + original[-1]
    full_array = np.empty((original.size + new_samples.size,), dtype=original.dtype)
    full_array[0::2] = original
    full_array[1::2] = new_samples
    return new_samples


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
    return ChoiSimulation.chi_final(tfinal)


def simple_noise_sampling(tfinal):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    ueV_conversion = 0.241799050402417
    sigma_test = 5.0 * ueV_conversion
    samples_init = 11
    x_init = np.linspace(-7 * sigma_test, 7 * sigma_test, samples_init)
    noise_samples_init = qmf.gaussian(x_init, 0.0, sigma_test)
    return None