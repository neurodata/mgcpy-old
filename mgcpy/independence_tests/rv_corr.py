import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial import distance_matrix


class RVCorr(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, option=False):
        """
        :param data_matrix_X: is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array

        :param data_matrix_Y: is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array

        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()

        :param option: a boolean indicating wether the test will be RV correlation or CCa correlation
        :type compute_distance_matrix: FunctionType or callable()
        """
        IndependenceTest.__init__(
            self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.option = option

    def test_statistic(self, data_matrix_X, data_matrix_Y):
        """
        Computes the RV/CCa correlation measure between two datasets.
        - Directly computes linear correlation for RV
        - Calculates local linear correlations for CCa

        :param data_matrix_X: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array

        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array

        :return: returns a list of two items, that contains:
            - :test_statistic: the sample test statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following key:
                    - :covar: a 2D matrix of all covariances [-1,1]

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.rv_corr import RVCorr

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> rvcorr = RVCorr(X, Y, None)
        >>> rvcorr_stat, rvcorr_covar = rvcorr.test_statistic()
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
        
        independence_test_metadata = {"covariance": covar}
        
        return corr, independence_test_metadata
