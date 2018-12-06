"""
    **Main MGC Independence Test Module**
"""

import time
import warnings

import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.mgc.local_correlation import local_correlations
from mgcpy.independence_tests.mgc.threshold_smooth import (smooth_significant_local_correlations,
                                                           threshold_local_correlations)


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

    def test_statistic(self, matrix_X, matrix_Y):
        """
        Computes the MGC measure between two datasets.

            - It first computes all the local correlations
            - Then, it returns the maximal statistic among all local correlations based on thresholding.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a square matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a square matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

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

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using MGC and permutation test.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a square matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a square matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

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
        return super(MGC, self).p_value(matrix_X, matrix_Y)
