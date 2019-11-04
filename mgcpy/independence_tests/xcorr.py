import numpy as np
from scipy.stats import chi2

from mgcpy.independence_tests.ts_abstract_class import TimeSeriesIndependenceTest


class LjungBoxX(TimeSeriesIndependenceTest):
    def __init__(self, max_lag=1):
        self.max_lag = max_lag
        self.which_test = "LjungBoxX"

    def test_statistic(self, matrix_X, matrix_Y):
        """
        Test statistic for LjungBox between two time series.

        :param matrix_X: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_Y: 2D `numpy.array`
        """
        matrix_X, matrix_Y, M, n = self._validate_input(
            matrix_X, matrix_Y, self.max_lag
        )

        cross_corrs = self._compute_cross_corr(matrix_X, matrix_Y)

        q_stat = 0
        for i, j in enumerate(range(1, self.max_lag + 1)):
            q_stat += cross_corrs[i] / (n - j)

        q_stat *= n * (n + 2)

        self.test_statistic_ = q_stat

    def p_value(self, matrix_X, matrix_Y):
        """
        P-value for LjungBox between two time series.

        :param matrix_X: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_Y: 2D `numpy.array`
        """
        self.test_statistic(matrix_X, matrix_Y)

        self.p_value_ = chi2.sf(self.test_statistic_, self.max_lag)

        return self.p_value_

    def _validate_input(self, matrix_X, matrix_Y, M):
        """
        Helper function to validate inputs.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param M: Max lag for test statistic.
        :type M: integer

        :return: returns a list of four items, that contains:

            - :matrix_X: ``[n*n]` distance matrix X.
            - :matrix_Y: ``[n*n]` distance matrix Y.
            - :block_size: Block size for block permutation procedure
            - :n: The sample size.
        :rtype: list
        """
        assert (
            matrix_X.shape[0] == matrix_Y.shape[0]
        ), "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be different from q"
        n = matrix_X.shape[0]
        if M >= n - 4:
            raise ValueError("max_lag must be less than n - 4.")

        # Represent univariate data as matrices.
        # Use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not.
        if len(matrix_X.shape) == 1:
            matrix_X = matrix_X.reshape((n, 1))
        if len(matrix_Y.shape) == 1:
            matrix_Y = matrix_Y.reshape((n, 1))

        M = self.max_lag if self.max_lag is not None else np.ceil(np.log(n))

        return matrix_X, matrix_Y, M, n

    def _compute_cross_corr(self, X, Y):
        """
        Note that what is returned is {Corr(X_{t}, Y_{t-1}), Corr(X_{t}, Y_{t-2}),...)}.

        :param matrix_X: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_Y: 2D `numpy.array`

        :return cross_corr: array of normalized cross-correlations up to max lag
        :rtype cross_corr: 1D `numpy.array`
        """
        X = X.ravel()
        Y = Y.ravel()

        covs = np.cov(X, Y)
        acvf_x = covs[0, 0]
        acvf_y = covs[1, 1]

        cross_corrs = []
        for j in range(1, self.max_lag + 1):
            covs = np.cov(X[j:], Y[:-j])
            ccvf_j = covs[0, 1] / np.sqrt(acvf_x * acvf_y)

            cross_corrs.append(ccvf_j)

        return cross_corrs
