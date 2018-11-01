import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.independence_tests.dcorr import DCorr


def compute_distance_matrix(data_matrix_X, data_matrix_Y):
    # obtain the pairwise distance matrix for X and Y
    dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
    dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
    return (dist_mtx_X, dist_mtx_Y)


def test_dcorr():
    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    mcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
    assert np.allclose(mcorr.test_statistic(), 0)

    # small simulated example of quadratic dependency: quad_sim(10, 2, noise=0, indep=False)
    X = np.array(
        [
            [0.95737163, -0.66346496],
            [0.71554598, -0.26426413],
            [0.68515833, -0.75692817],
            [-0.21809123, -0.44901085],
            [0.8511713, -0.76584218],
            [-0.55365671, -0.92794556],
            [-0.20974912, -0.47266052],
            [-0.42393941, -0.30563822],
            [0.08633873, 0.99153115],
            [-0.44444384, -0.99182435]])
    Y = np.array(
        [
            [0.39142434],
            [0.3403718],
            [0.09406136],
            [0.1958918],
            [0.21925826],
            [1.03556978],
            [0.19898681],
            [0.33265039],
            [0.33884542],
            [0.88426943]])
    mcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
    dcorr = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='dcorr')
    mantel = DCorr(data_matrix_X=X, data_matrix_Y=Y, compute_distance_matrix=compute_distance_matrix, corr_type='mantel')

    # test statistic
    assert np.allclose(mcorr.test_statistic(), 0.3117760199455171)
    assert np.allclose(dcorr.test_statistic(), 0.4454977629359435)
    assert np.allclose(mantel.test_statistic(), 0.2725479362090295)

    # test p value
    assert np.allclose(mcorr.p_value(), 0.03207910931266045)
    assert np.allclose(dcorr.p_value(), 0.07384, atol=0.1)
    assert np.allclose(mantel.p_value(), 0.25674, atol=0.1)
