import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr


class RVCorr(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, which_test='rv'):
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

        :param which_test: specifies which test to use, including 'rv','pearson','cca'.
        :type which_test: str
        """
        IndependenceTest.__init__(
            self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.which_test = which_test

    def test_statistic(self, data_matrix_X=None, data_matrix_Y=None):
        """
        Computes the Pearson/RV/CCa correlation measure between two datasets.
        - Default computes linear correlation for RV
        - Computes pearson's correlation for RV
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
            - :test_statistic: the sample test statistic
            - :independence_test_metadata: a ``dict`` of metadata with the following key:
                    - :covar: a 2D matrix of all covariances

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
        if data_matrix_X is None:
            data_matrix_X = self.data_matrix_X
        if data_matrix_Y is None:
            data_matrix_Y = self.data_matrix_Y

        row_X, columns_X = data_matrix_X.shape[0], data_matrix_X.shape[1]
        row_Y, columns_Y = data_matrix_Y.shape[0], data_matrix_Y.shape[1]

        mat1 = data_matrix_X - mb.repmat(np.mean(data_matrix_X, axis=0),
                                      data_matrix_X.shape[0], 1)
        mat2 = data_matrix_Y - mb.repmat(np.mean(data_matrix_Y, axis=0),
                                      data_matrix_Y.shape[0], 1)

        covar = np.dot(mat1.T, mat2)
        varX = np.dot(mat1.T, mat1)
        varY = np.dot(mat2.T, mat2)

        if (self.which_test == 'pearson') and ((row_X == 1 or columns_X == 1) and (row_Y == 1 or columns_Y == 1)):
            corr, covar = pearsonr(data_matrix_X, data_matrix_Y)
            corr, covar = corr[0], covar[0]
        elif (self.which_test == 'rv'):
            covar = np.trace(np.dot(covar, covar.T))
            corr = np.divide(covar, np.sqrt(np.trace(np.dot(varX, varX))
                                            * np.trace(np.dot(varY, varY))))
        else:
            if varX.size == 1 or varY.size == 1 or covar.size == 1:
                covar = np.sum(np.power(covar, 2))
                corr = np.divide(covar, np.sqrt(np.sum(np.power(varX, 2))
                                                * np.sum(np.power(varY, 2))))
            else:
                covar = np.sum(np.power(svds(covar, 1)[1], 2))
                corr = np.divide(covar, np.sqrt(np.sum(np.power(svds(varX, 1)[1], 2))
                                                * np.sum(np.power(svds(varY, 1)[1], 2))))

        independence_test_metadata = {"covariance": covar}

        return corr, independence_test_metadata