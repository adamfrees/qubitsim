# Module for noiseless evolution

import numpy as np
import CJFidelities

def time_evolution_RF(tfinal, arch='hybrid'):
    if arch == 'hybrid':
        import HybridQubit as hybrid
        indices = [0, 1]
        match_freq = 10.0
        operating_point = 3.5
        qubit = hybrid.SOSSHybrid(operating_point, match_freq)
        H0 = qubit.hamiltonian_lab()

        ChoiSimulation = CJFidelities.CJ(indices, H0, np.zeros((3,3)))
        if tfinal == 0:
            return ChoiSimulation.chi0
        else:
            return ChoiSimulation.chi_final_RF(tfinal)

if __name__ == '__main__':
    tsteps = 100
    trange = np.linspace(0, 30, tsteps)
    cj_time_array = np.zeros((9, 9, tsteps), dtype=complex)
    for i in range(tsteps):
        cj_time_array[:, :, i] += time_evolution_RF(trange[i])
    np.save('trange_noiseless_test.npy', trange)
    np.save('cj_array_noiseless_test.npy', cj_time_array)

