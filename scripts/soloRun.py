# This contains the scripts for an individual run for decoherence of 
# the quantum dot hybrid qubit

import math
import numpy as np

from context import qubitsim
from qubitsim.qubit import HybridQubit as hybrid
from qubitsim import CJFidelities as CJ

def norm_sin_integral(A, B, omega, exp_phi, sigma):
    """
    Return the integral
    int_{-infty}^{infty} exp(-x^2/(2 sigma^2)) (A e^(-i omega x + phi) + B) / sqrt(2 pi sigma^2).
    Inputs:
      A: oscillation amplitude,
      B: oscillation offset,
      omega: frequency of oscillation,
      exp_phi: effect of phase, e^(phi),
      sigma: standard deviation of quasi-static noise with gaussian weight
    Outputs:
      value of integral
    """
    return B + A * exp_phi * np.exp(-0.5 * sigma**2 * omega**2)


def noise_sample(qubit, ded, time):
    indices = [0, 1]
    H0 = np.diag(qubit.energies())
    noise = qubit.detuning_noise_qubit(ded)
    # noise = np.zeros((3,3))

    ChoiSimulation = CJ.CJ(indices, H0, noise)
    if time == 0:
        return ChoiSimulation.chi0
    else:
        return ChoiSimulation.chi_final_RF(time)

def fourier_find_freq(noise_samples, chi_array):
    """
    Find the peaks of the fft of the individual elements of the
    chi_array
    Inputs:
      noise_samples: array of sampled dipolar detuning noise values
      chi_array: array of resultant process matrices
        shape: (9, 9, noise_samples.shape[0])
    Outputs:
      peak_freq: frequency of the single mode input
    """
    dnoise = noise_samples[1] - noise_samples[0]
    chi_dim = chi_array.shape[0]
    freq = np.fft.fftfreq(noise_samples.shape[-1], d=dnoise)
    peak_freq = np.zeros((chi_dim, chi_dim))
    for i in range(chi_dim):
        for j in range(chi_dim):
            data_y = chi_array[i, j, :]
            sp = np.fft.fft(data_y)
            peaks = np.where(np.abs(sp) > 10.0 * np.mean(np.abs(sp)))
            if len(peaks) != 0:
                avgpeak = np.mean(np.abs(freq[peaks]))
                peak_freq[i, j] = avgpeak        
    return peak_freq


def process_chi_array(noise_samples, chi_array):
    noise_dim = chi_array.shape[-1]
    chi_dim = chi_array.shape[0]
    
    minNorm = np.empty((chi_dim, chi_dim))
    maxNorm = np.empty((chi_dim, chi_dim))
    zeroValue = np.empty((chi_dim, chi_dim), dtype=complex)
    offset = np.empty((chi_dim, chi_dim))
    for i in range(chi_dim):
        for j in range(chi_dim):
            minNorm[i, j] = np.min(np.abs(chi_array[i, j, :]))
            maxNorm[i, j] = np.max(np.abs(chi_array[i, j, :]))
            zeroValue[i, j] = chi_array[i, j, noise_dim // 2]
            offset[i, j] = np.mean(np.real(chi_array[i, j, :]))
    
    amplitude = maxNorm - minNorm
    expPhase = np.empty((chi_dim, chi_dim), dtype=complex)
    for i in range(chi_dim):
        for j in range(chi_dim):
            if (np.abs(amplitude[i, j]) <= 1e-8):
                expPhase[i, j] = 1.0
            else:
                expPhase[i, j] = (zeroValue[i, j] - offset[i, j]) / amplitude[i, j]

    peak_freq = fourier_find_freq(noise_samples, chi_array)  
    return amplitude, offset, peak_freq, expPhase


def average_process(qubit, time, sigma):
    """
    Generate array of process matrices dependent on dipolar 
    detuning noise
    Inputs:
      noise_samples: values of detuning noise in GHz
      time: time value since initialization in ns
    Outputs:
      cj_array: ((9, 9, len(noise_samples))) array of process matrices.
    """
    max_noise = 2.0 / (time)
    noise_samples = np.linspace(-max_noise, max_noise, 8193)
    noise_dim = noise_samples.shape[0]
    cj_array = np.zeros((9, 9, noise_dim), dtype=complex)
    for i in range(noise_dim):
        ded = noise_samples[i]
        cj_array[:, :, i] += noise_sample(qubit, ded, time)
    
    amplitude, offset, peakFreq, expPhase = process_chi_array(noise_samples, cj_array)
    return norm_sin_integral(amplitude, offset, peakFreq, expPhase, sigma)


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
    trange = generate_trange(tfinal)
    cj_array = np.empty((qubit.dim**2, qubit.dim**2, trange.shape[0]), dtype=complex)
    for i in range(trange.shape[0]):
        if trange[i] == 0:
            cj_array[:, :, i] += noise_sample(qubit, 0.0, 0.0)
        else:
            cj_array[:, :, i] += average_process(qubit, trange[i], sigma)
    return trange, cj_array


if __name__ == '__main__':
    params = {
        'ed_point': 1.0,
        'sigma' : 1.0,
        'delta1_var' : 1.0,
        'delta2_var' : 1.0
    }
    trange, cj_array = run_time_series(params)
    