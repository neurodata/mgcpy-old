import math
import warnings
from statistics import mean, stdev
from scipy.stats import norm, t

import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.utils.compute_distance_matrix import compute_distance
from mgcpy.independence_tests.utils.distance_transform import transform_distance_matrix

class DCorrX(IndependenceTest):

    def __init__(self, compute_distance_matrix=None, which_test='unbiased', max_lag=0):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()

        :param which_test: the type of distance covariance estimate to use, can be 'unbiased', 'biased' 'mantel'
        :type which_test: string

        :param max_lag: Maximum lead/lag to check for dependence between X_t and Y_t+j (M parameter)
        :type max_lag: int
        '''
        IndependenceTest.__init__(self)
        if which_test not in ['unbiased', 'biased']:
            raise ValueError('which_test must be unbiased or biased.')
        self.which_test = which_test
        self.dcorr = DCorr(which_test = self.which_test)
        self.max_lag = max_lag

    def test_statistic(self, matrix_X, matrix_Y, p = None):
        """
        Computes the (summed across lags) cross distance covariance estimate between two time series.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param p: bandwidth parameter for Bartlett Kernel.
        :type p: float

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample cdcv statistic (not necessarily within [-1,1])
            - :test_statistic_metadata: a ``dict`` of metadata with the following keys:
                    - :dist_mtx_X: the distance matrix of sample X
                    - :dist_mtx_Y: the distance matrix of sample X
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorr import DCorr
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> cdcv = CDCV(which_test = 'unbiased')
        >>> cdcv_statistic = cdcv.test_statistic(X, Y)
        """
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be different from q"
        if self.which_test == "unbiased" and matrix_X.shape[0] <= 3:
            raise ValueError('Cannot use unbiased estimator of distance covariance with n <= 3.')

        # Represent univariate data as matrices.
        # Use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not.
        n = matrix_X.shape[0]
        if len(matrix_X.shape) == 1:
            matrix_X = matrix_X.reshape((n,1))
        if len(matrix_Y.shape) == 1:
            matrix_Y = matrix_Y.reshape((n,1))
        matrix_X, matrix_Y = compute_distance(matrix_X, matrix_Y, self.compute_distance_matrix)

        M = self.max_lag if self.max_lag is not None else math.ceil(math.sqrt(n))
        dcorr = self.dcorr

        # Collect the test statistic by lag, and sum them for the full test statistic.
        dependence_by_lag = np.zeros(M+1)
        dcorr_statistic, _ = dcorr.test_statistic(matrix_X, matrix_Y)
        dependence_by_lag[0] = np.maximum(0.0, dcorr_statistic)

        # TO DO: parallelize?
        for j in range(1, M+1):
            dist_mtx_X = matrix_X[j:n,j:n]
            dist_mtx_Y = matrix_Y[0:(n-j),0:(n-j)]
            dcorr_statistic, _ = dcorr.test_statistic(dist_mtx_X, dist_mtx_Y)
            dependence_by_lag[j] = (n-j)*np.maximum(0.0, dcorr_statistic) / n

        # Reporting optimal lag
        optimal_lag = np.argmax(dependence_by_lag)
        test_statistic_metadata = { 'optimal_lag' : optimal_lag, 'dependence_by_lag' : dependence_by_lag }
        self.test_statistic_ = np.sum(dependence_by_lag)
        self.test_statistic_metadata_ = test_statistic_metadata
        return self.test_statistic_, test_statistic_metadata

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        '''
        Compute the p-value
        if the correlation test is unbiased, p-value can be computed using a t test
        otherwise computed using permutation test

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

        :return: p-value of distance correlation
        :rtype: numpy.float
        :return: returns a list of two items, that contains:

            - :p_value: ta ``numpy.float`` containing the p-value of the observed test statistic.
            - :p_value_metadata: a ``dict`` of metadata with the following keys:
                    - :null_distribution: the estimated (discrete) distribution of the test statistic
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorr import DCorr
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> cdcv = CDCV()
        >>> p_value, metadata = dcorr.p_value(X, Y, replication_factor = 100)
        '''
        return super(DCorrX, self).p_value_block(matrix_X, matrix_Y, replication_factor)
