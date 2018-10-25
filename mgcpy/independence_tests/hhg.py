import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial import distance_matrix


class HHG(IndependenceTest):
    """
    Calculates the HHG correlation statistic.

    :param data_matrix_X: an input distance matrix
    :param data_matrix_Y: an input distance matrix
    :param compute_distance_matrix: a function to compute the pairwise distance matrix
    :param option: a boolean indicating either that the test will be RV
                   correlation or CCA, defaults to True (RV)
    """

    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, option=True):
        IndependenceTest.__init__(
            self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.option = option

    def test_statistic(self):
        """
        Calculates the HHG test statistic

        :return: The local correlation ``corr`` and local covariance ``covar``
                 of the input data matricies
        """

        row_X, columns_X = self.data_matrix_X.shape[0], self.data_matrix_X.shape[1]
        row_Y, columns_Y = self.data_matrix_Y.shape[0], self.data_matrix_Y.shape[1]

        # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
        if row_X != columns_X or sum(self.data_matrix_X.diagonal()**2) > 0:
            dist_mtx_X = distance_matrix(
                self.data_matrix_X, self.data_matrix_X)
        else:
            dist_mtx_X = self.data_matrix_X
        if row_Y != columns_Y or sum(self.data_matrix_Y.diagonal()**2) > 0:
            dist_mtx_Y = distance_matrix(
                self.data_matrix_Y, self.data_matrix_Y)
        else:
            dist_mtx_Y = self.data_matrix_Y

        n = dist_mtx_X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    tmp1 = dist_mtx_X[i, :] <= dist_mtx_X[i, j]
                    tmp2 = dist_mtx_Y[i, :] <= dist_mtx_Y[i, j]
                    t11 = np.sum(tmp1 * tmp2) - 2
                    t12 = np.sum(tmp1 * (1-tmp2))
                    t21 = np.sum((1-tmp1) * tmp2)
                    t22 = np.sum((1-tmp1) * (1-tmp2))
                    denom = (t11+t12) * (t21+t22) * (t11+t21) * (t12+t22)
                    if denom > 0:
                        S[i, j] = (n-2) * \
                            np.power((t12*t21 - t11*t22), 2) / denom

        corr = np.sum(S)

        return corr
