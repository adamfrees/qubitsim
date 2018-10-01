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

    Gamma21 = (deriv1*sigma) / (math.sqrt(2))
    Gamma22 = (deriv2 * sigma**2) / (math.sqrt(2))
    Gamma23 = (deriv3 * sigma**3) / (math.sqrt(2))
    return 1.0 / np.sum(np.array([Gamma21, Gamma22, Gamma23]))


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
    trange1 = np.linspace(0, 0.2*tfinal, 100)
    trange2 = np.linspace(0.2*tfinal, tfinal, 100)
    trange = np.unique(np.concatenate((trange1, trange2)))
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
    