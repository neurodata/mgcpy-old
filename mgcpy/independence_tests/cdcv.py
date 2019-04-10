import math
import warnings
from statistics import mean, stdev
from scipy.stats import norm, t

import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.utils.compute_distance_matrix import compute_distance
from mgcpy.independence_tests.utils.distance_transform import transform_distance_matrix

class CDCV(IndependenceTest):

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
        self.max_lag = max_lag

    def test_statistic(self, matrix_X, matrix_Y, is_fast=False, fast_dcorr_data={}):
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

        :param is_fast: is a boolean flag which specifies if the test_statistic should be computed (approximated)
                        using the fast version of dcorr. This defaults to False.
        :type is_fast: boolean

        :param fast_dcorr_data: a ``dict`` of fast dcorr params, refer: self._fast_dcorr_test_statistic

            - :sub_samples: specifies the number of subsamples.
        :type fast_dcorr_data: dictonary

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
            raise ValueError('Cannot use unbiased estimator of CDCV with n <= 3.')

        # TO DO: fast CDCV.
        #if is_fast:
        #    test_statistic, test_statistic_metadata = self._fast_dcorr_test_statistic(matrix_X, matrix_Y, **fast_dcorr_data)
        #else:
        # Represent univariate data as matrices.
        # Use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not.
        n = matrix_X.shape[0]
        if len(matrix_X.shape) == 1:
            matrix_X = matrix_X.reshape((n,1))
        if len(matrix_Y.shape) == 1:
            matrix_Y = matrix_Y.reshape((n,1))
        matrix_X, matrix_Y = compute_distance(matrix_X, matrix_Y, self.compute_distance_matrix)

        # TO DO: parallelize.
        p = math.sqrt(n)
        M = self.max_lag if self.max_lag is not None else math.ceil(math.sqrt(n))
        bias_correct = 0 if self.which_test == 'biased' else 3

        # Collect the test statistic by lag, and sum them for the full test statistic.
        dependence_by_lag = np.zeros(M+1)
        dependence_by_lag[0] = np.maximum(0.0, self.cross_covariance_sum(matrix_X, matrix_Y))/(n-bias_correct)
        test_statistic = dependence_by_lag[0]
        for j in range(1,M+1):
            dist_mtx_X = matrix_X[j:n,j:n]
            dist_mtx_Y = matrix_Y[0:(n-j),0:(n-j)]
            dependence_by_lag[j] = ((1 - j/(p*(M+1)))**2)*(np.maximum(0.0, self.cross_covariance_sum(dist_mtx_X, dist_mtx_Y)))/(n-j-bias_correct)
            test_statistic += dependence_by_lag[j]

            # In asymmetric test, we do not add the following terms.
            # dist_mtx_X = matrix_X[0:(n-j),0:(n-j)]
            # dist_mtx_Y = matrix_Y[j:n,j:n]
            # test_statistic += ((1 - j/(p*(M+1)))**2)*(self.cross_covariance_sum(dist_mtx_X, dist_mtx_Y))/(n-j-bias_correct)

        # Reporting optimal lag
        optimal_lag = np.argmax(dependence_by_lag)
        test_statistic_metadata = { 'dist_mtx_X' : matrix_X,
                                    'dist_mtx_Y' : matrix_Y,
                                    'optimal_lag' : optimal_lag }
        self.test_statistic_ = test_statistic / ((M+1)*n)
        self.test_statistic_metadata_ = test_statistic_metadata
        return test_statistic, test_statistic_metadata

    """
    def _fast_dcorr_test_statistic(self, matrix_X, matrix_Y, sub_samples=10):
        '''
        Fast Dcor or Hsic test by subsampling that runs in O(ns*n), based on:
        Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic, “Large-scale kernel methods for independence testing,”
        Statistics and Computing, vol. 28, no. 1, pp. 113–130, 2018.

        Faster version of DCorr's test_statistic function

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param sub_samples: specifies the number of subsamples.
                            generally total_samples/sub_samples should be more than 4,
                            and ``sub_samples`` should be large than 10.
        :type sub_samples: integer

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample DCorr statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :sigma: computed standard deviation for computing the p-value next.
                    - :mu: computed mean for computing the p-value next.
        :rtype: list
        '''

        total_samples = matrix_Y.shape[0]
        num_samples = total_samples // sub_samples

        # if full data size (total_samples) is not more than 4 times of sub_samples, split to 4 samples
        # too few samples will fail the normal approximation and cause the test to be invalid

        if total_samples < 4 * sub_samples:
            sub_samples = total_samples // 4
            num_samples = 4

        # the observed statistics by subsampling
        test_statistic_sub_sampling = np.zeros(num_samples)

        # subsampling computation
        permuted_Y = matrix_Y
        for i in range(num_samples):
            sub_matrix_X = matrix_X[(sub_samples*i):sub_samples*(i+1), :]
            sub_matrix_Y = permuted_Y[(sub_samples*i):sub_samples*(i+1), :]

            test_statistic_sub_sampling[i], _ = self.test_statistic(sub_matrix_X, sub_matrix_Y)

        # approximate the null distribution by normal distribution
        sigma = stdev(test_statistic_sub_sampling) / math.sqrt(num_samples)
        mu = 0

        # compute the test statistic
        test_statistic = mean(test_statistic_sub_sampling)

        test_statistic_metadata = {"sigma": sigma,
                                   "mu": mu}

        return test_statistic, test_statistic_metadata
    """
    def cross_covariance_sum(self, dist_mtx_X, dist_mtx_Y):
        '''
        Helper function: Compute the sum of element-wise distances products
        Divide by n^2 to compute (biased) global covariance estimate.

        :param dist_mtx_X: a [(n-j)*(n-j)] distance matrix (lag j)
        :type dist_mtx_X: 2D numpy.array

        :param dist_mtx_Y: a [(n-j)*(n-j)] distance matrix (lag j)
        :type dist_mtx_Y: 2D numpy.array

        :return: the data covariance or variance based on the distance matrices
        :rtype: numpy.float
        '''

        transformed_distance_matrices = transform_distance_matrix(dist_mtx_X, dist_mtx_Y, base_global_correlation=self.which_test, is_ranked=False)
        transformed_dist_mtx_X = transformed_distance_matrices['centered_distance_matrix_A']
        transformed_dist_mtx_Y = transformed_distance_matrices['centered_distance_matrix_B']

        return np.sum(np.multiply(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_Y)))


    def p_value(self, matrix_X, matrix_Y, replication_factor=1000, is_fast=False, fast_dcorr_data={}):
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

        :param is_fast: is a boolean flag which specifies if the test_statistic should be computed (approximated)
                        using the fast version of dcorr. This defaults to False.
        :type is_fast: boolean

        :param fast_dcorr_data: a ``dict`` of fast dcorr params, refer: self._fast_dcorr_test_statistic

            - :sub_samples: specifies the number of subsamples.
        :type fast_dcorr_data: dictonary

        :return: p-value of distance correlation
        :rtype: numpy.float
        :return: returns a list of two items, that contains:

            - :p_value: ta ``numpy.float`` containing the p-value of the observed test statistic.
            - :p_value_metadata: a ``dict`` of metadata with the following keys:
                    - :test_stat_nulls: the estimated (discrete) distribution of the test statistic
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
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be equal to q"

        #if is_fast:
        #    p_value, p_value_metadata = self._fast_dcorr_p_value(matrix_X, matrix_Y, **fast_dcorr_data)
        #    self.p_value_ = p_value
        #    self.p_value_metadata_ = p_value_metadata
        #    return p_value, p_value_metadata
        #else:

        # Block bootstrap
        n = matrix_X.shape[0]
        block_size = int(np.ceil(np.sqrt(n)))
        test_statistic, test_statistic_metadata = self.test_statistic(matrix_X, matrix_Y)
        matrix_X = test_statistic_metadata['dist_mtx_X']
        matrix_Y = test_statistic_metadata['dist_mtx_Y']

        test_stats_null = np.zeros(replication_factor)
        for rep in range(replication_factor):
            # Generate new time series sample for Y
            permuted_indices = np.r_[[np.arange(t, t + block_size) for t in np.random.permutation((n // block_size) + 1)]].flatten()[:n]
            permuted_Y = matrix_Y[permuted_indices,:][:, permuted_indices] # TO DO: See if there is a better way to permute

            # Compute test statistic
            test_stats_null[rep], _ = self.test_statistic(matrix_X=matrix_X, matrix_Y=permuted_Y)

        p_value = np.where(test_stats_null >= test_statistic)[0].shape[0] / replication_factor
        p_value_metadata = {'test_stats_null' : test_stats_null}

        self.p_value_ = p_value
        self.p_value_metadata_ = p_value_metadata
        return p_value, p_value_metadata
    """
    def _fast_dcorr_p_value(self, matrix_X, matrix_Y, sub_samples=10):
        '''
        Fast Dcor or Hsic test by subsampling that runs in O(ns*n), based on:
        Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic, “Large-scale kernel methods for independence testing,”
        Statistics and Computing, vol. 28, no. 1, pp. 113–130, 2018.

        DCorr test statistic computation and permutation test by fast subsampling.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a ``[n*q]`` data matrix, a matrix with n samples in q dimensions
        :type matrix_Y: 2D numpy.array

        :param sub_samples: specifies the number of subsamples.
                            generally total_samples/sub_samples should be more than 4,
                            and ``sub_samples`` should be large than 10.
        :type sub_samples: integer

        :return: returns a list of two items, that contains:

            - :p_value: P-value of DCorr
            - :metadata: a ``dict`` of metadata with the following keys:

                    - :test_statistic: the sample DCorr statistic within ``[-1, 1]``
        :rtype: list
        '''
        test_statistic, test_statistic_metadata = self.test_statistic(matrix_X, matrix_Y, is_fast=True, fast_dcorr_data={"sub_samples": sub_samples})
        sigma = test_statistic_metadata["sigma"]
        mu = test_statistic_metadata["mu"]

        # compute p value
        p_value = 1 - norm.cdf(test_statistic, mu, sigma)

        # The results are not statistically significant
        if p_value > 0.05:
            warnings.warn("The p-value is greater than 0.05, implying that the results are not statistically significant.\n" +
                          "Use results such as test_statistic and optimal_scale, with caution!")

        p_value_metadata = {"test_statistic": test_statistic}

        return p_value, p_value_metadata
        """
