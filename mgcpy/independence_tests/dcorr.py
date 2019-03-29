import warnings
from statistics import mean

import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.utils.compute_distance_matrix import \
    compute_distance
from mgcpy.independence_tests.utils.distance_transform import \
    transform_distance_matrix
from mgcpy.independence_tests.utils.fast_functions import (_approx_null_dist,
                                                           _fast_pvalue,
                                                           _sample_atrr,
                                                           _sub_sample)


class DCorr(IndependenceTest):

    def __init__(self, compute_distance_matrix=None, which_test='unbiased', is_paired=False):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()

        :param which_test: the type of global correlation to use, can be 'unbiased', 'biased' 'mantel'
        :type which_test: string
        '''
        IndependenceTest.__init__(self)
        if which_test not in ['unbiased', 'biased', 'mantel']:
            raise ValueError('which_test must be unbiased, biased, or mantel')
        self.which_test = which_test
        self.is_paired = is_paired

    def test_statistic(self, matrix_X, matrix_Y, is_fast=False, fast_dcorr_data={}):
        """
        Computes the distance correlation between two datasets.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param is_fast: is a boolean flag which specifies if the test_statistic should be computed (approximated)
                        using the fast version of dcorr. This defaults to False.
        :type is_fast: boolean

        :param fast_dcorr_data: a ``dict`` of fast dcorr params, refer: self._fast_dcorr_test_statistic

            - :sub_samples: specifies the number of subsamples.
        :type fast_dcorr_data: dictonary

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample dcorr statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :variance_X: the variance of the data matrix X
                    - :variance_Y: the variance of the data matrix Y
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorr import DCorr
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> dcorr = DCorr(which_test = 'unbiased')
        >>> dcorr_statistic, test_statistic_metadata = dcorr.test_statistic(X, Y)
        """
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be equal to q"

        if is_fast:
            test_statistic, test_statistic_metadata = self._fast_dcorr_test_statistic(matrix_X, matrix_Y, **fast_dcorr_data)
        else:
            matrix_X, matrix_Y = compute_distance(matrix_X, matrix_Y, self.compute_distance_matrix)

            # perform distance transformation
            # transformed_dist_mtx_X, transformed_dist_mtx_Y = dist_transform(matrix_X, matrix_Y, self.which_test)

            transformed_distance_matrices = transform_distance_matrix(matrix_X, matrix_Y, base_global_correlation=self.which_test, is_ranked=False)
            transformed_dist_mtx_X = transformed_distance_matrices['centered_distance_matrix_A']
            transformed_dist_mtx_Y = transformed_distance_matrices['centered_distance_matrix_B']

            # transformed_dist_mtx need not be symmetric
            covariance = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_Y))
            variance_X = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_X))
            variance_Y = self.compute_global_covariance(transformed_dist_mtx_Y, np.transpose(transformed_dist_mtx_Y))

            # check the case when one of the dataset has zero variance
            if variance_X <= 0 or variance_Y <= 0:
                correlation = 0
            else:
                if self.is_paired:
                    n = transformed_dist_mtx_X.shape[0]
                    correlation = (variance_X/n/(n-1)) + (variance_Y/n/(n-1)) \
                        - 2*np.sum(np.multiply(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_Y)).diagonal())/n
                else:
                    correlation = covariance/np.real(np.sqrt(variance_X*variance_Y))

            # store the variance of X, variance of Y and the covariace as metadata
            test_statistic_metadata = {'variance_X': variance_X, 'variance_Y': variance_Y, 'covariance': covariance}

            # use absolute value for mantel coefficients

            if self.which_test == 'mantel':
                test_statistic = np.abs(correlation)
            else:
                test_statistic = correlation

        self.test_statistic_ = test_statistic
        self.test_statistic_metadata_ = test_statistic_metadata
        return test_statistic, test_statistic_metadata

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
        num_samples, sub_samples = _sample_atrr(matrix_Y, sub_samples)

        test_statistic_sub_sampling = _sub_sample(matrix_X, matrix_Y, self.test_statistic, num_samples, sub_samples, self.which_test)
        sigma, mu = _approx_null_dist(num_samples, test_statistic_sub_sampling, self.which_test)

        # compute the test statistic
        test_statistic = mean(test_statistic_sub_sampling)

        test_statistic_metadata = {"sigma": sigma,
                                   "mu": mu}

        return test_statistic, test_statistic_metadata

    def compute_global_covariance(self, dist_mtx_X, dist_mtx_Y):
        '''
        Helper function: Compute the global covariance using distance matrix A and B

        :param dist_mtx_X: a [n*n] distance matrix
        :type dist_mtx_X: 2D numpy.array

        :param dist_mtx_Y: a [n*n] distance matrix
        :type dist_mtx_Y: 2D numpy.array

        :return: the data covariance or variance based on the distance matrices
        :rtype: numpy.float
        '''
        return np.sum(np.multiply(dist_mtx_X, dist_mtx_Y))

    def unbiased_T(self, matrix_X, matrix_Y):
        '''
        Helper function: Compute the t-test statistic for unbiased dcorr

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :return: test statistic of t-test for unbiased dcorr
        :rtype: numpy.float
        '''
        test_stat, _ = self.test_statistic(matrix_X=matrix_X, matrix_Y=matrix_Y)

        n = matrix_X.shape[0]
        if n < 4:
            raise ValueError('Not enough samples, number of samples must be greater than 3')
            return None
        v = n*(n-3)/2
        # T converges in distribution to a t distribution under the null
        if test_stat == 1:
            '''
            if test statistic is 1, the t test statistic goes to inf
            '''
            T = np.inf
        else:
            T = np.sqrt(v-1) * test_stat / np.sqrt((1-np.square(test_stat)))
        return (T, v-1)

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

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorr import DCorr
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> dcorr = DCorr()
        >>> p_value, metadata = dcorr.p_value(X, Y, replication_factor = 100)
        '''
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be equal to q"

        if is_fast:
            p_value, p_value_metadata = self._fast_dcorr_p_value(matrix_X, matrix_Y, **fast_dcorr_data)
            self.p_value_ = p_value
            self.p_value_metadata_ = p_value_metadata
            return p_value, p_value_metadata
        else:
            return super(DCorr, self).p_value(matrix_X, matrix_Y)

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
        p_value = _fast_pvalue(test_statistic, test_statistic_metadata)

        # The results are not statistically significant
        if p_value > 0.05:
            warnings.warn("The p-value is greater than 0.05, implying that the results are not statistically significant.\n" +
                          "Use results such as test_statistic and optimal_scale, with caution!")

        p_value_metadata = {"test_statistic": test_statistic}

        return p_value, p_value_metadata
