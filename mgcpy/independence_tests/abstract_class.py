"""
    **Main Independence Test Abstract Class**
"""
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr, t
from mgcpy.independence_tests.utils.compute_distance_matrix import \
    compute_distance


def EUCLIDEAN_DISTANCE(x):
    return squareform(pdist(x, metric="euclidean"))


class IndependenceTest(ABC):
    """
    IndependenceTest abstract class

    Specifies the generic interface that must be implemented by
    all the independence tests in the mgcpy package.

    :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
    :type compute_distance_matrix: ``FunctionType`` or ``callable()``
    """

    def __init__(self, compute_distance_matrix=None):
        self.test_statistic_ = None
        self.test_statistic_metadata_ = None
        self.p_value_ = None
        self.p_value_metadata_ = None
        self.which_test = None

        if not compute_distance_matrix:
            compute_distance_matrix = EUCLIDEAN_DISTANCE
        self.compute_distance_matrix = compute_distance_matrix

        super().__init__()

    def get_name(self):
        """
        :return: the name of the independence test
        :rtype: string
        """
        return self.which_test

    @abstractmethod
    def test_statistic(self, matrix_X, matrix_Y):
        """
        Abstract method to compute the test statistic given two data matrices

        :param matrix_X: a ``[n*p]`` data matrix, a matrix with n samples in ``p`` dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a ``[n*q]`` data matrix, a matrix with n samples in ``q`` dimensions
        :type matrix_Y: 2D `numpy.array`

        :return: returns a list of two items, that contains:

            - :test_statistic_: the test statistic computed using the respective independence test
            - :test_statistic_metadata_: (optional) metadata other than the test_statistic,
                                        that the independence tests computes in the process
        :rtype: list
        """

        pass

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using the independence test and permutation test.

        :param matrix_X: a ``[n*p]`` matrix, a matrix with n samples in ``p`` dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a ``[n*q]`` matrix, a matrix with n samples in ``q`` dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

        :return: returns a list of two items, that contains:

            - :p_value_: P-value
            - :p_value_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                 that the independence tests computes in the process
        """
        # np.random.seed(int(time.time()))

        # calculte the test statistic with the given data
        test_statistic, independence_test_metadata = self.test_statistic(matrix_X, matrix_Y)

        if self.get_name() == "unbiased":
            '''
            for the unbiased centering scheme used to compute unbiased dcorr test statistic
            we can use a t-test to compute the p-value
            notation follows from: SzÃ©kely, GÃ¡bor J., and Maria L. Rizzo.
            "The distance correlation t-test of independence in high dimension."
            Journal of Multivariate Analysis 117 (2013): 193-213.
            '''
            null_distribution = []
            for _ in range(replication_factor):
                # use random permutations on the second data set
                premuted_matrix_Y = np.random.permutation(matrix_Y)

                temp_mgc_statistic, temp_independence_test_metadata = self.test_statistic(
                    matrix_X, premuted_matrix_Y)
                null_distribution.append(temp_mgc_statistic)

            T, df = self.unbiased_T(matrix_X=matrix_X, matrix_Y=matrix_Y)
            # p-value is the probability of obtaining values more extreme than the test statistic
            # under the null
            if T < 0:
                p_value = t.cdf(T, df=df)
            else:
                p_value = 1 - t.cdf(T, df=df)
            p_value_metadata = {"test_statistic": test_statistic,
                                "null_distribution": null_distribution}
        elif self.get_name() == "mgc":
            local_correlation_matrix = independence_test_metadata["local_correlation_matrix"]

            p_local_correlation_matrix = np.zeros(local_correlation_matrix.shape)
            p_value = 1/replication_factor

            null_distribution = []
            # compute sample MGC statistic and all local correlations for each set of permuted data
            for _ in range(replication_factor):

                # use random permutations on the second data set
                premuted_matrix_Y = np.random.permutation(matrix_Y)

                temp_mgc_statistic, temp_independence_test_metadata = self.test_statistic(
                        matrix_X, premuted_matrix_Y)
                null_distribution.append(temp_mgc_statistic)
                temp_local_correlation_matrix = temp_independence_test_metadata["local_correlation_matrix"]

                p_value += ((temp_mgc_statistic >= test_statistic) * (1/replication_factor))
                p_local_correlation_matrix += ((temp_local_correlation_matrix >=
                                                local_correlation_matrix) * (1/replication_factor))

            p_value_metadata = {"test_statistic": test_statistic,
                                "null_distribution": null_distribution,
                                "p_local_correlation_matrix": p_local_correlation_matrix,
                                "local_correlation_matrix": local_correlation_matrix,
                                "optimal_scale": independence_test_metadata["optimal_scale"]}
        elif self.get_name() == "kendall":
            test_statistic, p_value = kendalltau(matrix_X, matrix_Y)
            p_value_metadata = {"test_statistic": test_statistic}
        elif self.get_name() == "spearman":
            test_statistic, p_value = spearmanr(matrix_X, matrix_Y)
            p_value_metadata = {"test_statistic": test_statistic}
        elif self.get_name() == "pearson":
            test_statistic, p_value = pearsonr(matrix_X.reshape(-1), matrix_Y.reshape(-1))
            p_value_metadata = {"test_statistic": test_statistic}
        else:
            # estimate the null by a permutation test
            test_stats_null = np.zeros(replication_factor)
            for rep in range(replication_factor):
                permuted_x = np.random.permutation(matrix_X)
                permuted_y = np.random.permutation(matrix_Y)
                test_stats_null[rep], _ = self.test_statistic(matrix_X=permuted_x, matrix_Y=permuted_y)
            test_stats_null[0] = test_statistic
            # p-value is the probability of observing more extreme test statistic under the null
            p_value = np.where(test_stats_null >= test_statistic)[0].shape[0] / replication_factor
            p_value_metadata = {"test_statistic": test_statistic,
                                "null_distribution": test_stats_null}

        # Correct for a p_value of 0. This is because, with bootstrapping permutations, a value of 0 is not valid
        if p_value == 0:
            p_value = 1 / replication_factor
        self.p_value_ = p_value
        self.p_value_metadata_ = p_value_metadata
        return p_value, p_value_metadata

    def p_value_block(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using block permutation test.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

        :return: returns a list of two items, that contains:

            - :p_value: P-value of MGC
            - :metadata: a ``dict`` of metadata with the following keys:
                    - :null_distribution: numpy array representing distribution of test statistic under null.
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.mgc.mgc_ts import MGC_TS
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> mgc_ts = MGC_TS()
        >>> p_value, metadata = mgc_ts.p_value(X, Y, replication_factor = 100)
        """
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be equal to q"

        # Compute test statistic
        n = matrix_X.shape[0]
        if len(matrix_X.shape) == 1:
            matrix_X = matrix_X.reshape((n, 1))
        if len(matrix_Y.shape) == 1:
            matrix_Y = matrix_Y.reshape((n, 1))
        matrix_X, matrix_Y = compute_distance(matrix_X, matrix_Y, self.compute_distance_matrix)
        test_statistic, test_statistic_metadata = self.test_statistic(matrix_X, matrix_Y)

        # Block bootstrap
        block_size = int(np.ceil(np.sqrt(n)))
        test_stats_null = np.zeros(replication_factor)
        for rep in range(replication_factor):
            # Generate new time series sample for Y
            permuted_indices = np.r_[[np.arange(t, t + block_size) for t in np.random.choice(n, n // block_size + 1)]].flatten()[:n]
            permuted_indices = np.mod(permuted_indices, n)
            permuted_Y = matrix_Y[np.ix_(permuted_indices, permuted_indices)]

            # Compute test statistic
            test_stats_null[rep], _ = self.test_statistic(matrix_X, permuted_Y)

        self.p_value_ = np.sum(np.greater(test_stats_null, test_statistic)) / replication_factor
        if self.p_value == 0.0:
            self.p_value = 1 / replication_factor
        self.p_value_metadata_ = {'null_distribution': test_stats_null}

        return self.p_value_, self.p_value_metadata_
