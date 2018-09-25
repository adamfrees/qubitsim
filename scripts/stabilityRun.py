#!/usr/bin/env python

# Script to run a series of stability runs
# Will take an input job index to select the correct
# parameters

import sys
import math

import numpy as np

from soloRun import run_time_series


def package_files(step, params, trange, process_array):
    """Package the information into a single .npz file 
    to be output"""
    ed = np.array(params['ed_point'])
    sigma = np.array(params['sigma'])
    delta1 = np.array(params['delta1_var'])
    delta2 = np.array(params['delta2_var'])
    filename = '{0:02d}output.npz'.format(step)
    np.savez(filename,
             ed=ed,
             sigma=sigma,
             delta1=delta1,
             delta2=delta2,
             trange=trange,
             process_array=process_array)
    return None


def runJob(job_index):
    """Run the job specified by the job_index"""
    operating_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    delta_var = np.linspace(0.9, 1.1, 21)
    ueV_conversion = 0.241799050402417
    sigma_array = np.array([1.0, 5.0, 10.0]) * ueV_conversion

    param_array = np.array(
        np.meshgrid(
            delta_var, delta_var, sigma_array, operating_points,
            indexing='ij'
            )).T.reshape(
                (operating_points.shape[0] * sigma_array.shape[0] * delta_var.shape[0]**2, 4))

    start_index = job_index * delta_var.shape[0]**2
    for step in range(21*21): 
        local_params = {
            'ed_point' : param_array[start_index + step][3],
            'sigma' : param_array[start_index + step][2],
            'delta1_var': param_array[start_index + step][1],
            'delta2_var': param_array[start_index + step][0]
        }
        trange, process_over_time = run_time_series(local_params)
        package_files(step, local_params, trange, process_over_time)
    return None



if __name__ == "__main__":
    job_index = int(sys.argv[1])
    runJob(job_index)
    