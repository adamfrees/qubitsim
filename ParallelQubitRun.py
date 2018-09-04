from mpi4py import MPI
import math
import os
import numpy as np

import OperatingPointSelection as trun

def SaveFiles(index_array, ed_point, delta1_point, delta2_point,
              trange, sigma_array, mass_cj_array):
    root_folder = os.path.join('Data', 'run0')
    file_name = os.path.join(root_folder,
                             '{:02d}{:02d}{:02d}'.format(
                                 index_array[0], index_array[1], index_array[2]))
    np.savez(file_name,
             op_point = ed_point,
             delta1_var = delta1_point,
             delta2_var = delta2_point,
             trange = trange,
             sigma_array = sigma_array,
             mass_chi_array = mass_chi_array
             )


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None

operating_point_array = np.linspace(0.5, 5.5, size)
delta1_array = np.linspace(0.95, 1.05, 11)
delta2_array = np.linspace(0.95, 1.05, 11)

if rank == 0:
    data0 = np.array(range(size), dtype=int)
    data1 = np.array(range(11), dtype=int)
    data2 = np.array(range(11), dtype=int)
    data = np.ravel(np.array(np.meshgrid(data0, data1, data2)).T)

recvbuf = np.empty((3*11*11), dtype=int)
comm.Scatter(data, recvbuf, root=0)

operating_points = recvbuf.reshape((11*11, 3))
for i in range(11*11):
    index_array = operating_points[i, :]
    ed_point = operating_point_array[index_array[0]]
    delta1_point = delta1_array[index_array[1]]
    delta2_point = delta2_array[index_array[2]]
    trange, sigma_array, mass_chi_array = trun.time_evolution_point(ed_point, delta1_point, delta2_point)
    SaveFiles(index_array, ed_point, delta1_point, delta2_point, 
              trange, sigma_array, mass_chi_array)
