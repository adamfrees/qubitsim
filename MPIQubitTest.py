from mpi4py import MPI
import math
import numpy as np

import HybridQubit as hybrid

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None
if rank == 0:
    data = np.linspace(0.5, 5.5, 3*size)

recvbuf = np.empty((3), dtype=float)
comm.Scatter(data, recvbuf, root=0)

for i in range(len(recvbuf)):
    qubit = hybrid.SOSSHybrid(recvbuf[i], 10.0)
    print('The rank of the worker is {}'.format(rank))
    print('The sent detuning point is {}'.format(recvbuf[i]))
    print('The qubit detuning is {}'.format(qubit.ed / qubit.stsplitting))