

import numpy as np
from scipy.spatial.distance import pdist, squareform

from mgcpy.independence_tests.rv_corr import RVCorr

def compute_distance_matrix(data_matrix_X, data_matrix_Y):
    # obtain the pairwise distance matrix for X and Y
    dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
    dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
    return (dist_mtx_X, dist_mtx_Y)

def test_local_corr():
    rvcorr = RVCorr()
    a = np.array([1, 4, 6, 5, 1, 9, 12, 3])
    b = np.arange(8)
    rvcorr = RVCorr(a, b, compute_distance_matrix)
    rvcorr2 = RVCorr(a, b, compute_distance_matrix, option=1)
    assert rvcorr.test_statistic() == 1
    assert rvcorr2.test_statistic() == 1