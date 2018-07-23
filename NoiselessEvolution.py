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

        ChoiSimulation
