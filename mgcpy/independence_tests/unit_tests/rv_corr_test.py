import numpy as np
from mgcpy.independence_tests.rv_corr import RVCorr
from scipy.spatial.distance import pdist, squareform


def compute_distance_matrix(data_matrix_X, data_matrix_Y):
    # obtain the pairwise distance matrix for X and Y
    dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
    dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
    return (dist_mtx_X, dist_mtx_Y)


def test_local_corr():
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504,
                  1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917,
                  7.1748, 6.5792, 2.4012])[:, np.newaxis]
    rvcorr = RVCorr(X, Y, compute_distance_matrix)
    rvcorr2 = RVCorr(X, Y, compute_distance_matrix, option=1)
    test_stat1 = rvcorr.test_statistic()[0]
    test_stat2 = rvcorr2.test_statistic()[0]

    assert np.round(test_stat1, decimals=2) == 0.86
    assert np.round(test_stat2, decimals=2) == 0.89
