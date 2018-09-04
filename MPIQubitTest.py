from mpi4py import MPI
import math
import numpy as np

import HybridQubit as hybrid

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None
if rank == 0:
    data0 = np.linspace(0.5, 5.5, size)
    data1 = np.linspace(0.98, 1.02, 5)
    data2 = np.linspace(0.95, 1.05, 11)
    data = np.flatten(np.hstack((data0, data1, data2)))

recvbuf = np.empty((3), dtype=float)
comm.Scatter(data, recvbuf, root=0)

print(rank, recvbuf)