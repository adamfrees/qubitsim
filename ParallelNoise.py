# Parallel Noise Computations
# Uses parallel evalutions based on 
# each time step.
# Backbone formed from MPI4PY

from mpi4py import MPI
import numpy as np
import NoiseAveraging

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None

samples = 81

if rank == 0:
    trange = np.linspace(0, 30, size*20, dtype=float)
    send_trange = trange.reshape((20, size))
local_trange = np.empty((20), dtype=float)
local_cj_array = np.empty((9, 9, 20), dtype=complex)
comm.Scatterv(send_trange, local_trange, root=0)
for i in range(local_trange.shape[0]):
    cj_average, samples = NoiseAveraging.simple_noise_sampling(local_trange[i], samples)
    local_cj_array[:, :, i] += cj_average

if rank == 0:
    total_cj_array = np.empty((9, 9, 20, size), dtype=complex)

comm.Gather(local_cj_array, total_cj_array, root=0)
print(total_cj_array.reshape(9, 9, size*20))