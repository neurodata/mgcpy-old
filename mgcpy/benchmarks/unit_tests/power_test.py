import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.benchmarks.simulations import w_sim, ubern_sim
from mgcpy.benchmarks.power import power
from mgcpy.independence_tests.rv_corr import RVCorr


def test_power():
    # power of mcorr
    mcorr = DCorr(which_test='mcorr')
    mcorr_power = power(mcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mcorr_power, 0.673, atol=0.1)

    # power of dcorr
    dcorr = DCorr(which_test='dcorr')
    dcorr_power = power(dcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(dcorr_power, 0.863, atol=0.1)

    # power of mantel
    mantel = DCorr(which_test='mantel')
    mantel_power = power(mantel, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mantel_power, 0.993, atol=0.1)

    # power of pearson
    pearson = RVCorr(which_test='pearson')
    pearson_power = power(pearson, ubern_sim, num_samples=100, num_dimensions=1)
    assert np.allclose(pearson_power, 0.05688, atol=0.05)
