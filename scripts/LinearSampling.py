# This contains the scripts for an individual run for decoherence of 
# the quantum dot hybrid qubit

import math
import numpy as np

from context import qubitsim
from qubitsim.qubit import HybridQubit as hybrid
from qubitsim import CJFidelities as CJ


def noise_sample(qubit, ded, time):
    indices = [0, 1]
    H0 = np.diag(qubit.energies())
    noise = qubit.detuning_noise_qubit(ded)
    # noise = np.zeros((3,3))

    ChoiSimulation = CJ.CJ(indices, H0, noise)
    if time == 0:
        return 1.
    else:
        return ChoiSimulation.fidelity(time)


def average_fidelity(qubit, time, sigma):
    """
    Generate array of process matrices dependent on dipolar detuning noise
    Parameters
    ----------
    qubit: HybridQubit object
        the qubit to simulate under noise
    time: float
        time value since initialization
        Units: ns
    sigma: float
        standard deviation of quasistatic charge noise
        Units: GHz
   
    Returns
    -------
    weighted_average: ((9, 9)) complex array
        average process matrix under quasi-static noise
    """
    from scipy.stats import norm
    max_noise = 5.*sigma
    noise_samples = np.linspace(-max_noise, max_noise, 51) #51 samples is probably overkill
    weights = norm.pdf(noise_samples, 0.0, sigma)
    noise_dim = noise_samples.shape[0]
    fid_array = np.zeros((noise_dim))
    for i in range(noise_dim):
        ded = noise_samples[i]
        fid_array[i] += noise_sample(qubit, ded, time)

    norm = np.trapz(weights, x=noise_samples)
    weighted_average = np.trapz(weights * fid_array, x=noise_samples) / norm
    return weighted_average


def choosing_final_time(qubit, sigma):
    """ Function to make a guess at the final time required 
    for estimating decoherence"""

    deriv1 = np.abs(qubit.splitting_derivative(1))
    deriv2 = np.abs(qubit.splitting_derivative(2))
    deriv3 = np.abs(qubit.splitting_derivative(3))

    Gamma21 = (deriv1*sigma) / (math.sqrt(2))
    Gamma22 = (deriv2 * sigma**2) / (math.sqrt(2))
    Gamma23 = (deriv3 * sigma**3) / (math.sqrt(2))
    gamma_final_time = 1.0 / (np.sum(np.array([Gamma21, Gamma22, Gamma23])))
    
    if gamma_final_time > 5e5:
        return 5e5
    else:
        return gamma_final_time


def generate_trange(tmax):
    """
    Helper function to generate a correct time range
    Only worried about order of magnitude precision
    """
    subdim = 50
    max_exp = int(math.ceil(math.log10(tmax)))
    max_range = int(max_exp * subdim)
    trange = np.zeros((max_range))
    for i in range(1, max_exp+1):
        local_range = np.linspace(10**(i-1), 10**i, 10)
        trange[subdim*(i-1):subdim*(i)] = local_range
    return trange


def run_time_series(local_params):
    operating_point = local_params['ed_point']
    delta1_var = local_params['delta1_var']
    delta2_var = local_params['delta2_var']
    sigma = local_params['sigma']

    qubit = hybrid.SOSSHybrid(operating_point, 10.0)
    ed = qubit.ed
    stsplitting = qubit.stsplitting
    delta1 = qubit.delta1
    delta2 = qubit.delta2

    qubit = hybrid.HybridQubit(ed,
                               stsplitting,
                               delta1_var * delta1,
                               delta2_var * delta2)

    tfinal = choosing_final_time(qubit, sigma)
    trange = np.logspace(0,5,11)#generate_trange(tfinal) #I couldn't get this function to work - leaving it as a simple logspace for now.
    fid_time_array = np.zeros((trange.shape[0]))
    for i in range(trange.shape[0]):
        if trange[i] == 0:
            fid_time_array[i] += noise_sample(qubit, 0.0, 0.0)
        else:
            fid_time_array[i] += average_fidelity(qubit, trange[i], sigma)
    return trange, fid_time_array


if __name__ == '__main__':
    params = {
        'ed_point': 1.0,
        'sigma' : 1.0,
        'delta1_var' : 1.0,
        'delta2_var' : 1.0
    }
    trange, fid_array = run_time_series(params)
