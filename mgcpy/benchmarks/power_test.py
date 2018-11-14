import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.benchmarks.simulations import w_sim, ubern_sim
from mgcpy.benchmarks.power import power
from mgcpy.independence_tests.rv_corr import RVCorr


def compute_distance_matrix(data_matrix_X, data_matrix_Y):
    # obtain the pairwise distance matrix for X and Y
    dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
    dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
    return (dist_mtx_X, dist_mtx_Y)


def test_power():
    # power of mcorr
    mcorr = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
    mcorr_power = power(mcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mcorr_power, 0.673, atol=0.1)

    # power of dcorr
    dcorr = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix, corr_type='dcorr')
    dcorr_power = power(dcorr, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(dcorr_power, 0.863, atol=0.1)

    # power of mantel
    mantel = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix, corr_type='mantel')
    mantel_power = power(mantel, w_sim, num_samples=100, num_dimensions=3)
    assert np.allclose(mantel_power, 0.993, atol=0.1)

    # power of pearson
    pearson = RVCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix, which_test='pearson')
    pearson_power = power(pearson, ubern_sim, num_samples=100, num_dimensions=1)
    assert np.allclose(pearson_power, 0.05688, atol=0.05)

    # power for different simulations
    assert np.allclose(power(mcorr, sin_sim, simulation_type='sine_16pi'), 0.07307, atol=0.05)
    assert np.allclose(power(mcorr, multi_noise_sim, simulation_type='multi_noise'), 0.83968, atol=0.1)
    assert np.allclose(power(mcorr, multi_indept, simulation_type='multi_indept'), 0.05048, atol=0.05)
    assert np.allclose(power(mcorr, circle_sim, simulation_type='ellipse'), 0.8105, atol=0.1)
    assert np.allclose(power(mcorr, square_sim, simulation_type='diamond'), 0.19534, atol=0.1)
