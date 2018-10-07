import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.independence_tests.dcorr import DCorr


def test_dcorr():
    # small simulated example of exponential dependency
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504, 1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917, 7.1748, 6.5792, 2.4012])[:, np.newaxis]
    mcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
    dcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='dcorr')
    mantel = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mantel')

    assert np.allclose(mcorr.test_statistic(), 0.9706976134729944)
    assert np.allclose(dcorr.test_statistic(), 0.9440471713403025)
    assert np.allclose(mantel.test_statistic(), 0.8660255725921642)

    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    mcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
    assert np.allclose(mcorr.test_statistic(), 0)


def compute_distance_matrix(data_matrix_X, data_matrix_Y):
    # obtain the pairwise distance matrix for X and Y
    dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
    dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
    return (dist_mtx_X, dist_mtx_Y)
