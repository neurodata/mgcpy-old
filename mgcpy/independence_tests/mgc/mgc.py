import time
import numpy as np

from mgcpy.independence_tests.mgc.threshold_smooth import threshold_local_correlations, smooth_significant_local_correlations
from mgcpy.independence_tests.mgc.local_correlation import local_correlations
from mgcpy.independence_tests.abstract_class import IndependenceTest


class MGC(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, base_global_correlation='mgc'):
        '''
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

        :param base_global_correlation: specifies which global correlation to build up-on,
                                        including 'mgc','dcor','mantel', and 'rank'.
                                        Defaults to mgc.
        :type base_global_correlation: str
        '''

        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.base_global_correlation = base_global_correlation

    def test_statistic(self, data_matrix_X=None, data_matrix_Y=None):
        """
        Computes the MGC measure between two datasets.
        - It first computes all the local correlations
        - Then, it returns the maximal statistic among all local correlations based on thresholding.

        :param data_matrix_X: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array

        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array

        :return: returns a list of two items, that contains:
            - :test_statistic: the sample MGC statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :local_correlation_matrix: a 2D matrix of all local correlations within [-1,1]
                    - :optimal_scale: the estimated optimal scale as an [x, y] pair.

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.mgc.mgc import MGC

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> mgc = MGC(X, Y, None)
        >>> mgc_statistic, independence_test_metadata = mgc.test_statistic()
        """
        if data_matrix_X is None:
            data_matrix_X = self.data_matrix_X
        if data_matrix_Y is None:
            data_matrix_Y = self.data_matrix_Y

        # compute all local correlations
        local_correlation_matrix = local_correlations(data_matrix_X, data_matrix_Y, self.base_global_correlation)[
                                                      "local_correlation_matrix"]
        m, n = local_correlation_matrix.shape
        if m == 1 or n == 1:
            mgc_statistic = local_correlation_matrix[m - 1][n - 1]
            optimal_scale = m * n
        else:
            sample_size = len(data_matrix_X) - 1  # sample size minus 1

            # find a connected region of significant local correlations, by thresholding
            significant_connected_region = threshold_local_correlations(
                local_correlation_matrix, sample_size)

            # find the maximum within the significant region
            result = smooth_significant_local_correlations(
                significant_connected_region, local_correlation_matrix)
            mgc_statistic, optimal_scale = result["mgc_statistic"], result["optimal_scale"]

        independence_test_metadata = {"local_correlation_matrix": local_correlation_matrix,
                                      "optimal_scale": optimal_scale}
        return float('%.7f' % (mgc_statistic)), independence_test_metadata

    def p_value(self, replication_factor=1000):
        """
        Tests independence between two datasets using MGC and permutation test.

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: returns a list of two items, that contains:
            - :p_value: P-value of MGC
            - :metadata: a ``dict`` of metadata with the following keys:
                    - :test_statistic: the sample MGC statistic within [-1, 1]
                    - :p_local_correlation_matrix: a 2D matrix of the P-values of the local correlations
                    - :local_correlation_matrix: a 2D matrix of all local correlations within [-1,1]
                    - :optimal_scale: the estimated optimal scale as an [x, y] pair.

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.mgc.mgc import MGC

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> mgc = MGC(X, Y, None)
        >>> p_value, metadata = mgc.p_value(replication_factor = 100)
        """
        np.random.seed(int(time.time()))

        mgc_statistic, independence_test_metadata = self.test_statistic()
        local_correlation_matrix = independence_test_metadata["local_correlation_matrix"]

        p_local_correlation_matrix = np.zeros(local_correlation_matrix.shape)
        p_value = 0

        # compute sample MGC statistic and all local correlations for each set of permuted data
        for _ in range(replication_factor):
            # use random permutations on the second data set
            premuted_data_matrix_Y = np.random.permutation(self.data_matrix_Y)

            temp_mgc_statistic, temp_independence_test_metadata = self.test_statistic(
                self.data_matrix_X, premuted_data_matrix_Y)
            temp_local_correlation_matrix = temp_independence_test_metadata["local_correlation_matrix"]

            p_value += ((temp_mgc_statistic >= mgc_statistic) * (1/replication_factor))
            p_local_correlation_matrix += ((temp_local_correlation_matrix >=
                                            local_correlation_matrix) * (1/replication_factor))

        metadata = {"test_statistic": mgc_statistic,
                    "p_local_correlation_matrix": p_local_correlation_matrix,
                    "local_correlation_matrix": local_correlation_matrix,
                    "optimal_scale": independence_test_metadata["optimal_scale"]}

        return p_value, metadata
