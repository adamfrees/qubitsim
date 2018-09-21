from mpi4py import MPI
import math
import numpy as np

from qubitsim.qubit import HybridQubit as hybrid

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None
if rank == 0:
    data0 = np.linspace(0.5, 5.5, size, dtype=float)
    data1 = np.linspace(0.98, 1.02, 5, dtype=float)
    data2 = np.linspace(0.95, 1.05, 11, dtype=float)
    data = np.ravel(np.array(np.meshgrid(data0, data1, data2)).T)

recvbuf = np.empty((3*5*11), dtype=float)
comm.Scatter(data, recvbuf, root=0)
print(rank, recvbuf)