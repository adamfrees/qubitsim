# Testing functions for the file NoiseAveraging.py

import pytest
import numpy as np

import NoiseAveraging

@pytest.fixture
def noise_sample():
    import numpy as np
    orig = np.arange(0, 12, 2)
    interleave = np.arange(1, 11, 2)
    full = np.arange(0, 11, 1)
    return orig, interleave, full


class Test_Doubling_function(object):

    def test_noise_doubling(self, noise_sample):
        import numpy as np

        orig, interleave, full = noise_sample
        new_samples, full_sampling = NoiseAveraging.noise_doubling(orig)
        assert np.array_equal(new_samples, interleave)
        assert np.array_equal(full, full_sampling)