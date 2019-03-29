# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

"""
    **MGC's Distance Transform Module**
"""

import numpy as np
cimport numpy as np


cpdef dense_rank_data(np.ndarray[np.float_t, ndim=1] data_matrix):
    """
    Equivalent to scipy.stats.rankdata(x, "dense"), but faster!

    :param data_matrix: any data matrix.
    :type ranked_data_matrix: 2D numpy.array

    :return: dense ranked ``data_matrix``
    :rtype: 2D numpy.array
    """

    u, v = np.unique(data_matrix, return_inverse=True)
    return v + 1


cpdef rank_distance_matrix(np.ndarray[np.float_t, ndim=2] distance_matrix):
    """
    Sorts the entries within each column in ascending order

    For ties, the "minimum" ranking is used, e.g. if there are
    repeating distance entries, The order is like 1,2,2,3,3,4,...

    :param distance_matrix: a symmetric distance matrix.
    :type distance_matrix: 2D numpy.array

    :return: column-wise ranked matrix of ``distance_matrix``
    :rtype: 2D numpy.array
    """

    # faster than np.apply_along_axis
    return np.hstack([dense_rank_data(distance_matrix[:, i]).reshape(-1, 1) for i in range(distance_matrix.shape[0])])


cpdef center_distance_matrix(np.ndarray[np.float_t, ndim=2] distance_matrix, str base_global_correlation="mgc", is_ranked=True):
    """
    Appropriately transform distance matrices by centering them, based on the
    specified global correlation to build on

    :param distance_matrix: a symmetric distance matrix
    :type distance_matrix: 2D numpy.array

    :param base_global_correlation: specifies which global correlation to build up-on,
                                    including 'mgc','unbiased', 'biased', 'mantel', and 'rank'.
                                    Defaults to mgc.
    :type base_global_correlation: string

    :param is_ranked: specifies whether ranking within a column is computed or not
                      Defaults to True.
    :type is_ranked: boolean

    :return: A ``dict`` with the following keys:

            - :centered_distance_matrix: a ``[n*n]`` centered distance matrix
            - :ranked_distance_matrix: a ``[n*n]`` column-ranked distance matrix
    :rtype: dictionary
    """

    cdef int n = distance_matrix.shape[0]
    cdef np.ndarray ranked_distance_matrix = np.zeros((<object> distance_matrix).shape)

    if is_ranked:
        ranked_distance_matrix = rank_distance_matrix(distance_matrix)

    if base_global_correlation == "rank":
        distance_matrix = ranked_distance_matrix.astype(np.float)

    # 'mgc' distance transform (col-wise mean) - default
    cdef np.ndarray expected_distance_matrix = np.repeat(
        ((distance_matrix.mean(axis=0) * n) / (n-1)), n).reshape(-1, n).T

    # unbiased version of dcorr distance transform (col-wise mean + row-wise mean - mean)
    if base_global_correlation == "unbiased":
        expected_distance_matrix = np.repeat(((distance_matrix.mean(axis=0) * n) / (n-2)), n).reshape(-1, n).T \
                                    + np.repeat(((distance_matrix.mean(axis=1) * n) / (n-2)), n).reshape(-1, n) \
                                    - (distance_matrix.sum() / ((n-1) * (n-2)))

    # biased version of dcorr distance transform
    elif base_global_correlation == "biased":
        expected_distance_matrix = np.repeat(distance_matrix.mean(axis=0), n).reshape(-1, n).T \
                                  + np.repeat(distance_matrix.mean(axis=1), n).reshape(-1, n) \
                                  - (distance_matrix.sum() / (n * n))

    # mantel distance transform
    elif base_global_correlation == "mantel":
        expected_distance_matrix = np.array(distance_matrix.sum() / (n * (n-1)))

    cdef np.ndarray centered_distance_matrix = distance_matrix - expected_distance_matrix

    # the diagonal entries are excluded for unbiased and mgc centering, but not
    # excluded for biased and mantel(simple) centering. (from MGC Matlab)
    if base_global_correlation != "mantel" and base_global_correlation != "biased":
        np.fill_diagonal(centered_distance_matrix, 0)

    return {"centered_distance_matrix": centered_distance_matrix,
            "ranked_distance_matrix": ranked_distance_matrix}


cpdef transform_distance_matrix(np.ndarray[np.float_t, ndim=2] distance_matrix_A, np.ndarray[np.float_t, ndim=2] distance_matrix_B,
                                str base_global_correlation="mgc", is_ranked=True):
    """
    Transforms the distance matrices appropriately, with column-wise ranking if needed.

    :param distance_matrix_A: first symmetric distance matrix, ``[n*n]``
    :type distance_matrix: 2D numpy.array

    :param distance_matrix_B: second symmetric distance matrix, ``[n*n]``
    :type distance_matrix: 2D numpy.array

    :param base_global_correlation: specifies which global correlation to build up-on,
                                    including 'mgc','unbiased', 'biased', 'mantel', and 'rank'.
                                    Defaults to mgc.
    :type base_global_correlation: string

    :param is_ranked: specifies whether ranking within a column is computed or not,
                      if, base_global_correlation = "rank", then ranking is performed
                      regardless of the value if is_ranked. Defaults to True.
    :type is_ranked: boolean

    :return: A ``dict`` with the following keys:

            - :centered_distance_matrix_A: a ``[n*n]`` centered distance matrix of A
            - :centered_distance_matrix_B: a ``[n*n]`` centered distance matrix of B
            - :ranked_distance_matrix_A: a ``[n*n]`` column-ranked distance matrix of A
            - :ranked_distance_matrix_B: a ``[n*n]`` column-ranked distance matrix of B
    :rtype: dictionary


    **Example:**

    >>> import numpy as np
    >>> from scipy.spatial import distance_matrix
    >>> from mgcpy.mgc.distance_transform import transform_distance_matrix
    >>>
    >>> X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    >>> Y = np.array([[30, 20, 10], [5, 10, 20], [8, 16, 32]])
    >>> X_distance_matrix = distance_matrix(X, X)
    >>> Y_distance_matrix = distance_matrix(Y, Y)
    >>> transformed_distance_matrix_X_Y = transform_distance_matrix(X_distance_matrix, Y_distance_matrix)
    """

    if base_global_correlation == "rank":
        is_ranked = True

    transformed_distance_matrix_A = center_distance_matrix(
        distance_matrix_A, base_global_correlation, is_ranked)
    transformed_distance_matrix_B = center_distance_matrix(
        distance_matrix_B, base_global_correlation, is_ranked)

    transformed_distance_matrix = {"centered_distance_matrix_A": transformed_distance_matrix_A["centered_distance_matrix"],
                                   "centered_distance_matrix_B": transformed_distance_matrix_B["centered_distance_matrix"],
                                   "ranked_distance_matrix_A": transformed_distance_matrix_A["ranked_distance_matrix"],
                                   "ranked_distance_matrix_B": transformed_distance_matrix_B["ranked_distance_matrix"]}

    return transformed_distance_matrix
