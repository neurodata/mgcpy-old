import numpy as np
import scipy.stats


def rank_distance_matrix(distance_matrix):
    """"
    Sorts the entries within each column in ascending order

    For ties, the "minimum" ranking is used, e.g. if there are repeating distance entries,
    The order is like 1,2,2,3,3,4,...

    :param distance_matrix: a symmetric distance matrix.
    :return: column-wise ranked matrice of ``distance_matrix``
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
