"""
    **Main MGC Independence Test Module**
"""
import math
import warnings
from statistics import stdev

import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.mgc.local_correlation import local_correlations
from mgcpy.independence_tests.mgc.threshold_smooth import (smooth_significant_local_correlations,
                                                           threshold_local_correlations)
from scipy.stats import norm


class MGC(IndependenceTest):
    def __init__(self, compute_distance_matrix=None, base_global_correlation='mgc'):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: ``FunctionType`` or ``callable()``

        :param base_global_correlation: specifies which global correlation to build up-on,
                                        including 'mgc','dcor','mantel', and 'rank'.
                                        Defaults to mgc.
        :type base_global_correlation: string
        '''

        IndependenceTest.__init__(self, compute_distance_matrix)
        self.which_test = "mgc"
        self.base_global_correlation = base_global_correlation

    def test_statistic(self, matrix_X, matrix_Y, is_fast=False, fast_mgc_data={}):
        """
        Computes the MGC measure between two datasets.

            - It first computes all the local correlations
            - Then, it returns the maximal statistic among all local correlations based on thresholding.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :param is_fast: is a boolean flag which specifies if the test_statistic should be computed (approximated)
                        using the fast version of mgc. This defaults to False.
        :type is_fast: boolean

        :param fast_mgc_data: a ``dict`` of fast mgc params, refer: self._fast_mgc_test_statistic

            - :sub_samples: specifies the number of subsamples.
            - :null_only: specifies if subsampling is to be used for estimating the null only.
        :type fast_mgc_data: dictonary

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample MGC statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :local_correlation_matrix: a 2D matrix of all local correlations within ``[-1,1]``
                    - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.mgc.mgc import MGC
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> mgc = MGC()
        >>> mgc_statistic, test_statistic_metadata = mgc.test_statistic(X, Y)
        """
        if is_fast:
            mgc_statistic, test_statistic_metadata = self._fast_mgc_test_statistic(matrix_X, matrix_Y, **fast_mgc_data)
        else:
            # compute all local correlations
            distance_matrix_X = self.compute_distance_matrix(matrix_X)
            distance_matrix_Y = self.compute_distance_matrix(matrix_Y)
            local_correlation_matrix = local_correlations(distance_matrix_X, distance_matrix_Y,
                                                          base_global_correlation=self.base_global_correlation)["local_correlation_matrix"]
            m, n = local_correlation_matrix.shape
            if m == 1 or n == 1:
                mgc_statistic = local_correlation_matrix[m - 1][n - 1]
                optimal_scale = m * n
            else:
                sample_size = len(matrix_X) - 1  # sample size minus 1

                # find a connected region of significant local correlations, by thresholding
                significant_connected_region = threshold_local_correlations(
                    local_correlation_matrix, sample_size)

                # find the maximum within the significant region
                result = smooth_significant_local_correlations(
                    significant_connected_region, local_correlation_matrix)
                mgc_statistic, optimal_scale = result["mgc_statistic"], result["optimal_scale"]

            test_statistic_metadata = {"local_correlation_matrix": local_correlation_matrix,
                                       "optimal_scale": optimal_scale}

        self.test_statistic_ = mgc_statistic
        self.test_statistic_metadata_ = test_statistic_metadata
        return mgc_statistic, test_statistic_metadata

    def _fast_mgc_test_statistic(self, matrix_X, matrix_Y, sub_samples=100, null_only=True):
        """
        Faster version of MGC's test_statistic function

            - It computes local correlations and test statistics by subsampling
            - Then, it returns the maximal statistic among all local correlations based on thresholding.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :param sub_samples: specifies the number of subsamples.
                            generally total_samples/sub_samples should be more than 4,
                            and ``sub_samples`` should be large than 30.
        :type sub_samples: integer

        :param null_only: specifies if subsampling is to be used for estimating the null only OR to compute the observed statistic as well

            - *True:* uses subsampled statistics for estimating the null only and computes the observed statistic by full data,
                    this runs in ``O(total_samples^2 + sub_samples * total_samples)``
            - *False:* uses subsampled statistics for estimating the null and also computes the observed statistic by subsampling,
                     this runs in ``O(sub_samples*total_samples)``
        :type null_only: boolean

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample MGC statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :local_correlation_matrix: a 2D matrix of all local correlations within ``[-1,1]``
                    - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
                    - :sigma: computed standard deviation for computing the p-value next.
        :rtype: list
        """
        total_samples = matrix_Y.shape[0]
        num_samples = total_samples // sub_samples

        # if full data size (total_samples) is not more than 4 times of sub_samples, split to 4 samples
        # too few samples will fail the normal approximation and cause the test to be invalid

        if total_samples < 4 * sub_samples:
            sub_samples = total_samples // 4
            num_samples = 4

        # the observed statistics by subsampling
        test_statistic_sub_sampling = np.zeros(num_samples)

        # add trivial noise to break any ties
        matrix_X = matrix_X + 1e-10 * np.random.uniform(size=matrix_X.shape)
        matrix_Y = matrix_Y + 1e-10 * np.random.uniform(size=matrix_Y.shape)

        # the local correlations by subsampling
        local_correlation_matrix_sub_sampling = np.zeros((sub_samples, sub_samples, num_samples))

        # subsampling computation
        for i in range(num_samples):
            sub_matrix_X = matrix_X[(sub_samples*i):sub_samples*(i+1), :]
            sub_matrix_Y = matrix_Y[(sub_samples*i):sub_samples*(i+1), :]

            mgc_statistic, test_statistic_metadata = self.test_statistic(sub_matrix_X, sub_matrix_Y)
            test_statistic_sub_sampling[i], local_correlation_matrix_sub_sampling[:, :, i] = \
                mgc_statistic, test_statistic_metadata["local_correlation_matrix"]

        local_correlation_matrix = np.mean(local_correlation_matrix_sub_sampling, axis=2)
        sigma = stdev(test_statistic_sub_sampling) / np.sqrt(num_samples)

        sample_size = len(matrix_X) - 1  # sample size minus 1

        # find a connected region of significant local correlations, by thresholding
        significant_connected_region = threshold_local_correlations(
            local_correlation_matrix, sample_size)

        # find the maximum within the significant region
        result = smooth_significant_local_correlations(
            significant_connected_region, local_correlation_matrix)
        mgc_statistic, optimal_scale = result["mgc_statistic"], result["optimal_scale"]

        k, l = optimal_scale
        k = math.ceil((k - 1)/(sub_samples - 1) * (total_samples - 1)) + 1
        l = math.ceil((l - 1)/(sub_samples - 1) * (total_samples - 1)) + 1
        optimal_scale = [k, l]

        # the observed statistic uses the full data
        if null_only:
            mgc_statistic, test_statistic_metadata = self.test_statistic(matrix_X, matrix_Y)
            local_correlation_matrix = test_statistic_metadata["local_correlation_matrix"]
            optimal_scale = test_statistic_metadata["optimal_scale"]
            sigma /= math.sqrt(num_samples)

        test_statistic_metadata = {"local_correlation_matrix": local_correlation_matrix,
                                   "optimal_scale": optimal_scale,
                                   "sigma": sigma}

        return mgc_statistic, test_statistic_metadata

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000, is_fast=False, fast_mgc_data={}):
        """
        Tests independence between two datasets using MGC and permutation test.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

        :param is_fast: is a boolean flag which specifies if the p_value should be computed (approximated)
                        using the fast version of mgc. This defaults to False.
        :type is_fast: boolean

        :param fast_mgc_data: a ``dict`` of fast mgc params, , refer: self._fast_mgc_p_value

            - :sub_samples: specifies the number of subsamples.
            - :null_only: specifies if subsampling is to be used for estimating the null only.
            - :alpha: specifies the type 1 error level.
        :type fast_mgc_data: dictonary

        :return: returns a list of two items, that contains:

            - :p_value: P-value of MGC
            - :metadata: a ``dict`` of metadata with the following keys:

                    - :test_statistic: the sample MGC statistic within ``[-1, 1]``
                    - :p_local_correlation_matrix: a 2D matrix of the P-values of the local correlations
                    - :local_correlation_matrix: a 2D matrix of all local correlations within ``[-1,1]``
                    - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.mgc.mgc import MGC
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> mgc = MGC()
        >>> p_value, metadata = mgc.p_value(X, Y, replication_factor = 100)
        """
        if is_fast:
            p_value, p_value_metadata = self._fast_mgc_p_value(matrix_X, matrix_Y, **fast_mgc_data)
            self.p_value_ = p_value
            self.p_value_metadata_ = p_value_metadata
            return p_value, p_value_metadata
        else:
            return super(MGC, self).p_value(matrix_X, matrix_Y)

    def _fast_mgc_p_value(self, matrix_X, matrix_Y, sub_samples=100, null_only=True, alpha=0.01):
        '''
        Fast version of MGC's p-value computation

        MGC test statistic computation and permutation test by fast subsampling.
        Note that trivial amount of noise is added to matrix_X and matrix_Y,
        to break possible ties in data for MGC.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a ``[n*d]`` data matrix, a matrix with n samples in d dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a ``[n*d]`` data matrix, a matrix with n samples in d dimensions
        :type matrix_Y: 2D numpy.array

        :param sub_samples: specifies the number of subsamples.
                            generally total_samples/sub_samples should be more than 4,
                            and ``sub_samples`` should be large than 30.
        :type sub_samples: integer

        :param null_only: specifies if subsampling is to be used for estimating the null only OR to compute the observed statistic as well

            - *True:* uses subsampled statistics for estimating the null only and computes the observed statistic by full data,
                    this runs in ``O(total_samples^2 + sub_samples * total_samples)``
            - *False:* uses subsampled statistics for estimating the null and also computes the observed statistic by subsampling,
                     this runs in ``O(sub_samples*total_samples)``
        :type null_only: boolean

        :param alpha: specifies the type 1 error level.
                      this is is used to derive the confidence interval and estimate required sample size to achieve power 1.
        :type alpha: float

        :return: returns a list of two items, that contains:

            - :p_value: P-value of MGC
            - :metadata: a ``dict`` of metadata with the following keys:

                    - :test_statistic: the sample MGC statistic within ``[-1, 1]``
                    - :local_correlation_matrix: a 2D matrix of all local correlations within ``[-1,1]``
                    - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
                    - :confidence_interval: a ``[1*2]`` matrix representing the confidence_interval for the local correlation with 1-alpha confidence.
                    - :required_size: the required estimated sample size to have power 1 at level alpha
        :rtype: list
        '''
        mgc_statistic, test_statistic_metadata = self.test_statistic(matrix_X, matrix_Y, is_fast=True, fast_mgc_data={"sub_samples": sub_samples, "null_only": null_only})
        total_samples = matrix_Y.shape[0]
        sigma = test_statistic_metadata["sigma"]

        p_value = 1 - norm.cdf(mgc_statistic, 0, sigma)
        threshold = norm.ppf(1 - alpha, 0, 1)
        confidence_interval = [mgc_statistic - (threshold * sigma), mgc_statistic + (threshold * sigma)]

        if mgc_statistic < 0:
            required_size = np.Inf
        else:
            required_size = math.ceil(threshold * sigma * total_samples / mgc_statistic / 10) * 10

        # The results are not statistically significant
        if p_value > 0.05:
            warnings.warn("The p-value is greater than 0.05, implying that the results are not statistically significant.\n" +
                          "Use results such as test_statistic and optimal_scale, with caution!")

        p_value_metadata = {"test_statistic": mgc_statistic,
                            "local_correlation_matrix": test_statistic_metadata["local_correlation_matrix"],
                            "optimal_scale": test_statistic_metadata["optimal_scale"],
                            "confidence_interval": confidence_interval,
                            "required_size": required_size}

        return p_value, p_value_metadata
