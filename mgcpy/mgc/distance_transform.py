import numpy as np
import scipy.stats


def rank_distance_matrix(distance_matrix):
    """"
    Sorts the entries within each column in ascending order

    For ties, the "minimum" ranking is used, e.g. if there are
    repeating distance entries, The order is like 1,2,2,3,3,4,...

    :param distance_matrix: a symmetric distance matrix.
    :type distance_matrix: 2D numpy.array

    :return: column-wise ranked matrix of ``distance_matrix``
    :rtype: 2D numpy.array
    """
    n_rows = distance_matrix.shape[0]
    ranked_distance_matrix = np.zeros(distance_matrix.shape)
    for i in range(n_rows):
        column = distance_matrix[:, i]
        ranked_column = np.array(scipy.stats.rankdata(column, "min"))
        sorted_unique_ranked_column = sorted(list(set(ranked_column)))
        if (len(ranked_column) != len(sorted_unique_ranked_column)):
            for j, rank in enumerate(sorted_unique_ranked_column):
                ranked_column[ranked_column == rank] = j + 1
        ranked_distance_matrix[:, i] = ranked_column
    return ranked_distance_matrix


def center_distance_matrix(distance_matrix, base_global_correlation="mgc", is_ranked=True):
    """
    Appropriately transform distance matrices by centering them, based on the
    specified global correlation to build on

    :param distance_matrix: a symmetric distance matrix
    :type distance_matrix: 2D numpy.array

    :param base_global_correlation: specifies which global correlation to build up-on,
                                    including 'mgc','dcor','mantel', and 'rank'
    :type base_global_correlation: string

    :param is_ranked: specifies whether ranking within a column is computed or not
    :type is_ranked: boolean

    :return: dict(centered ``distance_matrix``, column-wise ranked ``distance_matrix``)
    :rtype: dictionary
    """
    n = distance_matrix.shape[0]
    ranked_distance_matrix = None
    expected_distance_matrix = None

    if is_ranked:
        ranked_distance_matrix = rank_distance_matrix(distance_matrix)
    else:
        ranked_distance_matrix = np.zeros(distance_matrix.shape)

    if base_global_correlation == "rank":
        distance_matrix = ranked_distance_matrix

    # 'mgc' distance transform (col-wise mean) - default
    if base_global_correlation == "mgc":
        expected_distance_matrix = np.repeat(((distance_matrix.mean(axis=0) * n) / (n-1)), n).reshape(-1, n).T

    # unbiased version of dcor distance transform (col-wise mean + row-wise mean - mean)
    elif base_global_correlation == "dcor":
        expected_distance_matrix = np.repeat(((distance_matrix.mean(axis=0) * n) / (n-2)), n).reshape(-1, n).T \
                                    + np.repeat(((distance_matrix.mean(axis=1) * n) / (n-2)), n).reshape(-1, n) \
                                    - (distance_matrix.sum() / ((n-1) * (n-2)))
        expected_distance_matrix += distance_matrix / n

    # mantel distance transform
    elif base_global_correlation == "mantel":
        expected_distance_matrix = distance_matrix.sum() / (n * (n-1))

    if not expected_distance_matrix:
        raise RuntimeError("Unknown base_global_correlation parameter: " + str(base_global_correlation))

    centered_distance_matrix = distance_matrix - expected_distance_matrix

    # the diagonal entries are always excluded
    centered_distance_matrix = np.fill_diagonal(centered_distance_matrix, 0)

    return {"centered_distance_matrix": centered_distance_matrix,
            "ranked_distance_matrix": ranked_distance_matrix}
