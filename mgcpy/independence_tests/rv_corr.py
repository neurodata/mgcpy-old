import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial import distance_matrix


class RVCorr(IndependenceTest):
    """
    Calculates the pearsons/Cca/rv correlation statistic.

    :param data_matrix_X: an input distance matrix
    :param data_matrix_Y: an input distance matrix
    :param compute_distance_matrix: a function to compute the pairwise distance matrix
    :param option: a boolean indicating either that the test will be Pearson's correlation or 
    """

    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, option=False):
        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.option = option

    def test_statistic(self):
        """
        Calculates all the local correlation coefficients.

        :return: The local correlation ``corr`` and local covariance ``covar``
                 of the input data matricies
        """

        row_X, columns_X = self.data_matrix_X.shape[0], self.data_matrix_X.shape[1]
        row_Y, columns_Y = self.data_matrix_Y.shape[0], self.data_matrix_Y.shape[1]

        # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
        if row_X != columns_X or sum(self.data_matrix_X.diagonal()**2) > 0:
            dist_mtx_X = distance_matrix(self.data_matrix_X, self.data_matrix_X)
        else:
            dist_mtx_X = self.data_matrix_X
        if row_Y != columns_Y or sum(self.data_matrix_Y.diagonal()**2) > 0:
            dist_mtx_Y = distance_matrix(self.data_matrix_Y, self.data_matrix_Y)
        else:
            dist_mtx_Y = self.data_matrix_Y

        mat1 = dist_mtx_X - mb.repmat(np.mean(dist_mtx_X, axis=0),
                                      dist_mtx_X.shape[0], 1)
        mat2 = dist_mtx_Y - mb.repmat(np.mean(dist_mtx_Y, axis=0),
                                      dist_mtx_Y.shape[0], 1)

        covar = np.matmul(a=mat1.T, b=mat2)
        varX = np.matmul(a=mat1.T, b=mat1)
        varY = np.matmul(a=mat2.T, b=mat2)

        self.option = np.minimum(np.abs(self.option), mat1.shape[1])
        if (self.option == 0):
            covar = np.trace(np.matmul(covar, covar.T))
            corr = np.divide(covar, np.sqrt(np.trace(np.matmul(varX, varX))
                                            * np.trace(np.matmul(varY, varY))))
        else:
            covar = np.sum(np.power(svds(covar, self.option)[1], 2))
            corr = np.divide(covar, np.sqrt(np.sum(np.power(svds(varX, self.option)[1], 2))
                                            * np.sum(np.power(svds(varY, self.option)[1], 2))))

        return corr, {"covariance": covar}
