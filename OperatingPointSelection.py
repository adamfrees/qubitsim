# Script to iterate over possible hybrid qubit operating locations 
# and determine the required tolerances for the tunnel couplings

import math
import numpy as np
import scipy.linalg as LA

import HybridQubit as hybrid
import NoiseAveraging as noise


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