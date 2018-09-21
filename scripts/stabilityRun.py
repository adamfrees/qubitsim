#!/usr/bin/env python

# Script to run a series of stability runs
# Will take an input job index to select the correct
# parameters

import sys
import math

import numpy as np

job_index = sys.argv[0]

operating_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
delta_var = np.linspace(0.9, 1.1, 21)
ueV_conversion = 0.241799050402417
sigma_array = np.array([1.0, 5.0, 10.0]) * ueV_conversion

param_array = np.array(
    np.meshgrid(
        operating_points, sigma_array, delta_var, delta_var
        )).T.reshape(
            (operating_points.shape[0] * sigma_array.shape[0] * delta_var.shape[0]**2, 4))

local_params = {
    'ed_point' : param_array[job_index][0],
    'sigma' : param_array[job_index][1],
    'delta1_var': param_array[job_index][2],
    'delta2_var': param_array[job_index][3]
}
trange, process_over_time = run_time_series(local_params)