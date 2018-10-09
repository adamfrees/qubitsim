#!/usr/bin/env python

# Script to run a series of stability runs
# Will take an input job index to select the correct
# parameters

import sys
import math

import numpy as np

from soloRun import run_time_series


def package_files(step, params, trange, process_array):
    """
    Package the information into a single .npz file 
    to be output
    """
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


def runMultivaryJob(job_index):
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
    for step in range(2*delta_var.shapep[0]): 
        local_params = {
            'ed_point' : param_array[start_index + step, 3],
            'sigma' : param_array[start_index + step, 2],
            'delta1_var': param_array[start_index + step, 1],
            'delta2_var': param_array[start_index + step, 0]
        }
        trange, process_over_time = run_time_series(local_params)
        package_files(step, local_params, trange, process_over_time)
    return None


def runSingleVaryJob(job_index):
    """
    Run the job specified by the job_index.
    These jobs will vary one tunnel coupling, 
    then vary the other tunnel coupling. Double variations
    are not considered.
    """
    operating_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    delta_var = np.linspace(0.9, 1.1, 21)
    ueV_conversion = 0.241799050402417
    sigma_array = np.array([1.0, 5.0, 10.0]) * ueV_conversion

    # Complicated way of translating the job_index to the 
    # appropriate operating point and noise value
    param_array1 = np.array(
        np.meshgrid(
            sigma_array, operating_points, indexing='ij')
            ).T.reshape((operating_points.shape[0]*sigma_array.shape[0], 2))

    # Create an array of tunnel coupling values to iterate over
    # Vary delta1, then vary delta2
    delta_array = np.zeros((2*delta_var.shape[0], 2))
    delta_array[:delta_var.shape[0], :] = np.array([delta_var, np.ones(delta_var.shape[0])]).T
    delta_array[delta_var.shape[0]:, :] = np.array([np.ones(delta_var.shape[0]), delta_var]).T

    for step in range(2*delta_var.shape[0]): 
        local_params = {
            'ed_point' : param_array1[job_index, 1],
            'sigma' : param_array1[job_index, 0],
            'delta1_var': delta_array[step, 0],
            'delta2_var': delta_array[step, 1]
        }
        trange, process_over_time = run_time_series(local_params)
        package_files(step, local_params, trange, process_over_time)
    return None


def atomistic_job(job_index):
    """
    Run a single ed, sigma, delta1, delta2 setting for the qubit
    """
    operating_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    delta_var = np.linspace(0.95, 1.05, 11)
    ueV_conversion = 0.241799050402417
    sigma_array = np.array([1.0, 5.0, 10.0]) * ueV_conversion

    op_index = job_index % (len(sigma_array) * len(delta_var)**2)
    new_index = job_index - op_index*(len(sigma_array) * len(delta_var)**2)
    sigma_index = new_index % (len(delta_var)**2)
    new_index = new_index - sigma_index * (len(delta_var)**2)
    delta1_index = new_index % len(delta_var)
    new_index = new_index - delta1_index*len(delta_var)
    delta2_index = new_index

    local_params = {
        'ed_point' : operating_points[op_index],
        'sigma' :    sigma_array[sigma_index],
        'delta1_var': delta_var[delta1_index],
        'delta2_var': delta_var[delta2_index]
    }
    trange, process_over_time = run_time_series(local_params)
    package_files(job_index, local_params, trange, process_over_time)
    return None

def runSingleTestJob(job_index):
    """
    Run a single job with ideal tunnel couplings
    """
    operating_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ueV_conversion = 0.241799050402417
    sigma = 5.0 * ueV_conversion
    # sigma_array = np.array([1.0, 5.0, 10.0]) * ueV_conversion

    local_params = {
        'ed_point' : operating_points[job_index],
        'sigma' : sigma,
        'delta1_var': 1.0,
        'delta2_var': 1.0
        }
    trange, process_over_time = run_time_series(local_params)
    package_files(job_index, local_params, trange, process_over_time)
    return None




if __name__ == "__main__":
    job_index = int(sys.argv[1])
    atomistic_job(job_index)
    