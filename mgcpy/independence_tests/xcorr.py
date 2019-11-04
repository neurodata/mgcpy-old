import numpy as np
from scipy.stats import chi2, norm

from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.ts_abstract_class import TimeSeriesIndependenceTest


class LjungBoxX(TimeSeriesIndependenceTest):
    def __init__(self, max_lag=1):
        self.max_lag = max_lag
        self.which_test = "LjungBoxX"

    def p_value(self, matrix_X, matrix_Y):
        """"
        Test statistic for MGCX between two time series.

        :param matrix_X: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*1] data matrix, a matrix with n samples in 1 dimensions
        :type matrix_Y: 2D `numpy.array`
        """
        self.test_statistic(matrix_X, matrix_Y)

        return self.p_value_

    def test_statistic(self, matrix_X, matrix_Y):
        matrix_X, matrix_Y, M, n = self._validate_input(
            matrix_X, matrix_Y, self.max_lag
        )

        cross_corrs = self._compute_cross_corr(matrix_X, matrix_Y)
        q_stat, p_value = self._compute_q_stat(cross_corrs, n)

        self.test_statistic_ = q_stat
        self.p_value_ = p_value

        return self.test_statistic_

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
        n = X.shape[0]

        X0 = X - X.mean()
        Y0 = Y - Y.mean()

        d = np.correlate(np.ones(n), np.ones(n), "full")

        cross_cov = np.correlate(X0.ravel(), Y0.ravel(), "full") / d
        cross_corr = cross_cov / (np.std(X) * np.std(Y))

        return cross_corr[: n - 1][::-1][: self.max_lag + 1]

    def _compute_q_stat(self, corrs, n_obs):
        q_stat = 0
        for i, k in enumerate(range(1, len(corrs) + 1)):
            q_stat += corrs[i] / (n_obs - k)
        q_stat *= n_obs * (n_obs + 2)

        p_value = chi2.sf(q_stat, self.max_lag)

        return q_stat, p_value
