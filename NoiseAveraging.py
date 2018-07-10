# Calculating Noise-averaged time evolution 
# using Choi-Jamiolkowski isomorphisms

import math
import numpy as np

import QMFormulas as qmf
import HybridQubit as hybrid
import CJFidelities as cj


def noise_sample_run(ded, tfinal):
    qubit = hybrid.SOSSHybrid(3.0, 10.0)
    H0 = qubit.hamiltonian_lab()
    noise = qubit.detuning_noise_lab(ded)
    ChoiSimulation = cj.CJ([0,1], H0+noise)
    return ChoiSimulation.chi_final(tfinal)


def simple_noise_sampling(tfinal):
    """
    This algorithm will start with
    a coarse noise sample and will progressively 
    refine the sample until convergence is found
    """
    ueV_conversion = .241799050402417
    sigma_test = 5.0 * ueV_conversion
    samples_init = 11
    x_init = np.linspace(-7 * sigma_test, 7 * sigma_test, samples_init)
    noise_samples_init = qmf.gaussian(x_init, 0.0, sigma_test)
    return None