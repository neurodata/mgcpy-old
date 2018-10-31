import numpy as np


cpdef local_covariance(distance_matrix_A, distance_matrix_B, ranked_distance_matrix_A, ranked_distance_matrix_B):
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

    cdef int n = distance_matrix_A.shape[0]
    cdef int n_X = np.max(ranked_distance_matrix_A) + 1
    cdef int n_Y = np.max(ranked_distance_matrix_B) + 1
    covariance_X_Y = np.zeros((n_X, n_Y))
    expected_X = np.zeros(n_X)
    expected_Y = np.zeros(n_Y)

    # summing up the the element-wise product of A and B based on the ranks,
    # yields the local family of covariances
    cdef float a, b
    cdef int i, j, k, l
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
