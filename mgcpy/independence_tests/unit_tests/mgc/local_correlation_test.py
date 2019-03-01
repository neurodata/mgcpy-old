import numpy as np
import pytest

from mgcpy.independence_tests.mgc_utils.local_correlation import \
    local_correlations


def test_local_correlations():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]], dtype=np.float)
    Y = np.array([[30, 20, 10], [5, 10, 20], [8, 16, 32]], dtype=np.float)

    local_correlation_matrix = np.array([[0, 0, 0.0000000],
                                         [0, 1, 1.0000000],
                                         [0, 1, 0.9905307]])
    local_variance_A = np.array([0.000, 2874.478, 3698.165])
    local_variance_B = np.array([0.0000, 97.4382, 135.4389])

    result = local_correlations(X, Y)

    assert np.allclose(local_correlation_matrix, result["local_correlation_matrix"])
    assert np.allclose(local_variance_A, result["local_variance_A"])
    assert np.allclose(local_variance_B, result["local_variance_B"])
