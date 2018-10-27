"""
    MGCPY Local Correlations
"""

import numpy as np
from scipy.spatial import distance_matrix
import warnings

from mgcpy.independence_tests.mgc.distance_transform import transform_distance_matrix
from mgcpy.independence_tests.mgc.local_cov import local_covariance_cython


def local_covariance(distance_matrix_A, distance_matrix_B, ranked_distance_matrix_A, ranked_distance_matrix_B):
    """
    Computes all local covariances simultaneously in O(n^2).

    :param distance_matrix_A: first distance matrix (centered or appropriately transformed), [n*n]
    :type distance_matrix_A: 2D numpy.array

    :param distance_matrix_B: second distance matrix (centered or appropriately transformed), [n*n]
    :type distance_matrix_B: 2D numpy.array

    :param ranked_distance_matrix_A: column-wise ranked matrix of ``A``, [n*n]
    :type ranked_distance_matrix_A: 2D numpy.array

    :param ranked_distance_matrix_B: column-wise ranked matrix of ``B``, [n*n]
    :type ranked_distance_matrix_B: 2D numpy.array

    :return: matrix of all local covariances, [n*n]
    :rtype: 2D numpy.array
    """
    # convert float32 numpy array to int, as it will be used as array indices [0 to n-1]
    ranked_distance_matrix_A = ranked_distance_matrix_A.astype(np.int) - 1
    ranked_distance_matrix_B = ranked_distance_matrix_B.astype(np.int) - 1

    n = distance_matrix_A.shape[0]
    n_X = np.max(ranked_distance_matrix_A) + 1
    n_Y = np.max(ranked_distance_matrix_B) + 1
    covariance_X_Y = np.zeros((n_X, n_Y))
    expected_X = np.zeros(n_X)
    expected_Y = np.zeros(n_Y)

    # summing up the the element-wise product of A and B based on the ranks,
    # yields the local family of covariances
    for i in range(n):
        for j in range(n):
            a = distance_matrix_A[i, j]
            b = distance_matrix_B[i, j]
            k = ranked_distance_matrix_A[i, j]
            l = ranked_distance_matrix_B[i, j]

            covariance_X_Y[k, l] += a * b

            expected_X[k] += a
            expected_Y[l] += b

    covariance_X_Y[:, 0] = np.cumsum(covariance_X_Y[:, 0])
    expected_X = np.cumsum(expected_X)

    covariance_X_Y[0, :] = np.cumsum(covariance_X_Y[0, :])
    expected_Y = np.cumsum(expected_Y)

    for k in range(n_X - 1):
        for l in range(n_Y - 1):
            covariance_X_Y[k+1, l+1] += (covariance_X_Y[k+1, l] +
                                         covariance_X_Y[k, l+1] - covariance_X_Y[k, l])

    # centering the covariances
    covariance_X_Y = covariance_X_Y - ((expected_X.reshape(-1, 1) @ expected_Y.reshape(-1, 1).T) / n**2)  # caveat when porting from R (reshape)

    return covariance_X_Y


def local_correlations(matrix_A, matrix_B, base_global_correlation="mgc"):
    """
    Computes all the local correlation coefficients in O(n^2 log n)

    :param matrix_A: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type matrix_A: 2D numpy.array

    :param matrix_B: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type matrix_B: 2D numpy.array

    :param base_global_correlation: specifies which global correlation to build up-on,
                                    including 'mgc','dcor','mantel', and 'rank'.
                                    Defaults to mgc.
    :type base_global_correlation: str

    :return: A ``dict`` with the following keys:
    :rtype: dict
        - :local_correlation_matrix: a 2D matrix of all local correlations within [-1,1]
        - :local_variance_A: all local variances of A
        - :local_variance_B: all local variances of B

    **Example:**
    >>> import numpy as np
    >>> from scipy.spatial import distance_matrix
    >>> from mgcpy.mgc.local_correlation import local_correlations

    >>> X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    >>> Y = np.array([[30, 20, 10], [5, 10, 20], [8, 16, 32]])
    >>> result = local_correlations(X, Y)
    """

    # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
    if matrix_A.shape[0] != matrix_A.shape[1] or sum(matrix_A.diagonal()**2) > 0:
        matrix_A = distance_matrix(matrix_A, matrix_A)
    if matrix_B.shape[0] != matrix_B.shape[1] or sum(matrix_B.diagonal()**2) > 0:
        matrix_B = distance_matrix(matrix_B, matrix_B)

    transformed_result = transform_distance_matrix(matrix_A, matrix_B, base_global_correlation)

    # compute all local covariances
    local_covariance_matrix = local_covariance_cython(
        transformed_result["centered_distance_matrix_A"],
        transformed_result["centered_distance_matrix_B"].T,
        transformed_result["ranked_distance_matrix_A"],
        transformed_result["ranked_distance_matrix_B"].T)

    # compute local variances for data A
    local_variance_A = local_covariance_cython(
        transformed_result["centered_distance_matrix_A"],
        transformed_result["centered_distance_matrix_A"].T,
        transformed_result["ranked_distance_matrix_A"],
        transformed_result["ranked_distance_matrix_A"].T)
    local_variance_A = local_variance_A.diagonal()

    # compute local variances for data B
    local_variance_B = local_covariance_cython(
        transformed_result["centered_distance_matrix_B"],
        transformed_result["centered_distance_matrix_B"].T,
        transformed_result["ranked_distance_matrix_B"],
        transformed_result["ranked_distance_matrix_B"].T)
    local_variance_B = local_variance_B.diagonal()

    warnings.filterwarnings("ignore")
    # normalizing the covariances yields the local family of correlations
    local_correlation_matrix = local_covariance_matrix / \
        np.sqrt(local_variance_A.reshape(-1, 1) @ local_variance_B.reshape(-1, 1).T).real  # 2 caveats when porting from R (np.sqrt and reshape)
    # avoid computational issues that may cause a few local correlations to be negligebly larger than 1
    local_correlation_matrix[local_correlation_matrix > 1] = 1
    warnings.filterwarnings("default")

    # set any local correlation to 0 if any corresponding local variance is less than or equal to 0
    local_correlation_matrix[local_variance_A <= 0, :] = 0
    local_correlation_matrix[:, local_variance_B <= 0] = 0

    return {"local_correlation_matrix": local_correlation_matrix,
            "local_variance_A": local_variance_A,
            "local_variance_B": local_variance_B}
