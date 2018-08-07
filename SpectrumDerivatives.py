# Some code to find the optimal step for calculating 
# derivatives of the hybrid qubit spectrum

import math
import numpy as np
import matplotlib.pyplot as plt

import HybridQubit as hybrid

def centered_difference_ed():
    coeff1 = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    coeff2 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    coeff3 = np.array([1/8, -1, 13/8, 0, -13/8, 1, -1/8])

    h_test_array = np.logspace(-8, -1, 100)

    base_qubit = hybrid.SOSSHybrid(0.5, 10.0)
    ed_base = base_qubit.ed
    stsplitting_base = base_qubit.stsplitting
    delta1_base = base_qubit.delta1
    delta2_base = base_qubit.delta2

    deriv1_array = np.zeros((100))
    deriv2_array = np.zeros((100))
    deriv3_array = np.zeros((100))

    for i in range(100):
        h = h_test_array[i]
        qm3 = hybrid.HybridQubit(ed_base - 3 * h, stsplitting_base, delta1_base, delta2_base)
        qm2 = hybrid.HybridQubit(ed_base - 2 * h, stsplitting_base, delta1_base, delta2_base)
        qm1 = hybrid.HybridQubit(ed_base - 1 * h, stsplitting_base, delta1_base, delta2_base)
        qp1 = hybrid.HybridQubit(ed_base + 1 * h, stsplitting_base, delta1_base, delta2_base)
        qp2 = hybrid.HybridQubit(ed_base + 2 * h, stsplitting_base, delta1_base, delta2_base)
        qp3 = hybrid.HybridQubit(ed_base + 3 * h, stsplitting_base, delta1_base, delta2_base)

        splitting_array = np.array([qm3.qubit_splitting(), qm2.qubit_splitting(),
                                    qm1.qubit_splitting(),
                                    base_qubit.qubit_splitting(),
                                    qp1.qubit_splitting(),
                                    qp2.qubit_splitting(), qp3.qubit_splitting()]) / (2*math.pi)
        
        deriv1_array[i] = np.dot(splitting_array, coeff1)
        deriv2_array[i] = np.dot(splitting_array, coeff2)
        deriv3_array[i] = np.dot(splitting_array, coeff3)

    fig, ax = plt.subplots()
    ax.loglog(h_test_array, np.abs(deriv1_array))
    ax.loglog(h_test_array, np.abs(deriv2_array))
    ax.loglog(h_test_array, np.abs(deriv3_array))
    ax.set_xlabel(r'step size, $h$')
    plt.show()


def centered_difference_delta1():
    coeff1 = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    coeff2 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    coeff3 = np.array([1/8, -1, 13/8, 0, -13/8, 1, -1/8])
    
    h_test_array = np.logspace(-12, -1, 100)

    base_qubit = hybrid.SOSSHybrid(1.0, 10.0)
    ed_base = base_qubit.ed
    stsplitting_base = base_qubit.stsplitting
    delta1_base = base_qubit.delta1
    delta2_base = base_qubit.delta2

    deriv1_array = np.zeros((100))
    deriv2_array = np.zeros((100))
    deriv3_array = np.zeros((100))

    for i in range(100):
        h = h_test_array[i]
        qm3 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base - 3*h, delta2_base)
        qm2 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base - 2*h, delta2_base)
        qm1 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base - h, delta2_base)
        qp1 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base + h, delta2_base)
        qp2 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base + 2*h, delta2_base)
        qp3 = hybrid.HybridQubit(ed_base, stsplitting_base, delta1_base + 3*h, delta2_base)

        splitting_array = np.array([qm3.qubit_splitting(), qm2.qubit_splitting(),
                                    qm1.qubit_splitting(),
                                    base_qubit.qubit_splitting(),
                                    qp1.qubit_splitting(),
                                    qp2.qubit_splitting(), qp3.qubit_splitting()]) / (2*math.pi)
        
        deriv1_array[i] = np.dot(splitting_array, coeff1)
        deriv2_array[i] = np.dot(splitting_array, coeff2)
        deriv3_array[i] = np.dot(splitting_array, coeff3)

    fig, ax = plt.subplots()
    ax.loglog(h_test_array, np.abs(deriv1_array))
    ax.loglog(h_test_array, np.abs(deriv2_array))
    ax.loglog(h_test_array, np.abs(deriv3_array))
    ax.set_xlabel(r'step size, $h$')
    plt.show()


if __name__ == '__main__':
    centered_difference_delta1()