import numpy as np
import pytest
from mgcpy.independence_tests.manova import Manova
from mgcpy.benchmarks.hypothesis_tests.three_sample_test.power import generate_three_two_d_gaussians
from mgcpy.hypothesis_tests.transforms import k_sample_transform


def test_local_corr():
    np.random.seed(0)
    matrix_X, matrix_Y, matrix_Z = generate_three_two_d_gaussians(2, 100, 3)

    data = np.concatenate([matrix_X, matrix_Y, matrix_Z], axis=0)
    labels = np.concatenate([np.repeat(1, matrix_X.shape[0]), np.repeat(2, matrix_Y.shape[0]), np.repeat(3, matrix_Z.shape[0])], axis=0).reshape(-1, 1)

    matrix_U, matrix_V = k_sample_transform(data, labels, is_y_categorical=True)

    # Against linear simulations
    manova = Manova()
    test_stat = manova.test_statistic(matrix_U, matrix_V)[0]

    assert manova.get_name() == 'manova'
    assert np.allclose(test_stat, 0.06, atol=1.e-2)
