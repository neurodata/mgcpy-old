import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr


class RVCorr(IndependenceTest):
    def __init__(self, compute_distance_matrix=None, which_test='rv'):
        """
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()

        :param which_test: specifies which test to use, including 'rv', 'pearson', and 'cca'.
        :type which_test: str
        """
        IndependenceTest.__init__(self, compute_distance_matrix)
        self.which_test = which_test

    def test_statistic(self, matrix_X=None, matrix_Y=None):
        """
        Computes the Pearson/RV/CCa correlation measure between two datasets.

        - Default computes linear correlation for RV
        - Computes pearson's correlation
        - Calculates local linear correlations for CCa

        :param matrix_X: a [n*p] data matrix, a matrix with n samples in p dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*q] data matrix, a matrix with n samples in q dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: returns a list of two items, that contains:

            - :test_statistic_: test statistic
            - :test_statistic_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                         that the independence tests computes in the process
        :rtype: float, dict

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.rv_corr import RVCorr

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> rvcorr = RVCorr()
        >>> rvcorr_test_stat = rvcorr.test_statistic(X, Y)
        """
        row_X, columns_X = matrix_X.shape[0], matrix_X.shape[1]
        row_Y, columns_Y = matrix_Y.shape[0], matrix_Y.shape[1]

        mat1 = matrix_X - mb.repmat(np.mean(matrix_X, axis=0),
                                    matrix_X.shape[0], 1)
        mat2 = matrix_Y - mb.repmat(np.mean(matrix_Y, axis=0),
                                    matrix_Y.shape[0], 1)

        covar = np.dot(mat1.T, mat2)
        varX = np.dot(mat1.T, mat1)
        varY = np.dot(mat2.T, mat2)

        if (self.which_test == 'pearson') and ((row_X == 1 or columns_X == 1) and (row_Y == 1 or columns_Y == 1)):
            corr, covar = pearsonr(matrix_X, matrix_Y)
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
        self.test_statistic_ = corr
        self.test_statistic_metadata_ = {"covariance": covar}

        return self.test_statistic_, self.test_statistic_metadata_

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using the independence test.

        :param matrix_X: a [n*p] data matrix, a matrix with n samples in p dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*q] data matrix, a matrix with n samples in q dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: returns a list of two items, that contains:

            - :p_value_: P-value
            - :p_value_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                 that the independence tests computes in the process
        :rtype: float, dict

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.rv_corr import RVCorr

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> rvcorr = RVCorr()
        >>> rvcorr_p_value = rvcorr.p_value(X, Y)
        """
        return super(RVCorr, self).p_value(matrix_X, matrix_Y)
