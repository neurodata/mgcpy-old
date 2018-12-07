import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.benchmarks.simulations import w_sim, ubern_sim, sin_sim, multi_noise_sim, multi_indep_sim, circle_sim, square_sim
from mgcpy.benchmarks.power import power, power_given_data
from mgcpy.independence_tests.rv_corr import RVCorr
import pickle

'''
def test_power():
    # power of mcorr
    mcorr = DCorr(which_test='unbiased')
    mcorr_power = power(mcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mcorr_power, 0.673, atol=0.1)

    # power of dcorr
    dcorr = DCorr(which_test='biased')
    dcorr_power = power(dcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(dcorr_power, 0.69, atol=0.1)

    # power of mantel
    mantel = DCorr(which_test='mantel')
    mantel_power = power(mantel, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mantel_power, 0.993, atol=0.1)

    # power of pearson
    pearson = RVCorr(which_test='pearson')
    pearson_power = power(pearson, ubern_sim, num_samples=100, num_dimensions=1)
    assert np.allclose(pearson_power, 0.05688, atol=0.05)

    # power for different simulations
    assert np.allclose(power(mcorr, sin_sim, simulation_type='sine_16pi'), 0.07307, atol=0.1)
    assert np.allclose(power(mcorr, multi_noise_sim, simulation_type='multi_noise'), 0.83968, atol=0.1)
    assert np.allclose(power(mcorr, multi_indep_sim, simulation_type='multi_indept'), 0.05048, atol=0.1)
    assert np.allclose(power(mcorr, circle_sim, simulation_type='ellipse'), 0.764, atol=0.1)
    assert np.allclose(power(mcorr, square_sim, simulation_type='diamond'), 0.19534, atol=0.1)


def test_power_given_data():
    sim = 4
    dim = 10
    mcorr = DCorr(which_test='unbiased')
    dcorr = DCorr(which_test='biased')
    mantel = DCorr(which_test='mantel')
    tests = [mcorr, dcorr, mantel]

    for test in tests:
        estimated_power = np.zeros(dim)
        for d in range(1, dim+1):
            estimated_power[d-1] = power_given_data(test, sim, num_samples=100, num_dimensions=d)
        file = open('./mgcpy/benchmarks/power_curves_dimensions/{}_{}_dimensions.pkl'.format(sim, test.get_name()), 'rb')
        pickle.load(file, true_power)
        assert np.allclose(estimated_power, true_power)
'''
