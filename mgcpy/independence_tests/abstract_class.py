"""
    **Main Independence Test Abstract Class**
"""
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr, t


def EUCLIDEAN_DISTANCE(x): return squareform(pdist(x, metric='euclidean'))


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
        '''
        :return: the name of the independence test
        :rtype: string
        '''
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
        np.random.seed(int(time.time()))

        # calculte the test statistic with the given data
        test_statistic, independence_test_metadata = self.test_statistic(matrix_X, matrix_Y)

        if self.get_name() == "unbiased":
            '''
            for the unbiased centering scheme used to compute unbiased dcorr test statistic
            we can use a t-test to compute the p-value
            notation follows from: Székely, Gábor J., and Maria L. Rizzo.
            "The distance correlation t-test of independence in high dimension."
            Journal of Multivariate Analysis 117 (2013): 193-213.
            '''
            T, df = self.unbiased_T(matrix_X=matrix_X, matrix_Y=matrix_Y)
            # p-value is the probability of obtaining values more extreme than the test statistic
            # under the null
            if T < 0:
                p_value = t.cdf(T, df=df)
            else:
                p_value = 1 - t.cdf(T, df=df)
            p_value_metadata = {}
        elif self.get_name() == "mgc":
            local_correlation_matrix = independence_test_metadata["local_correlation_matrix"]

            p_local_correlation_matrix = np.zeros(local_correlation_matrix.shape)
            p_value = 0

            # compute sample MGC statistic and all local correlations for each set of permuted data
            for _ in range(replication_factor):
                # use random permutations on the second data set
                premuted_matrix_Y = np.random.permutation(matrix_Y)

                temp_mgc_statistic, temp_independence_test_metadata = self.test_statistic(
                    matrix_X, premuted_matrix_Y)
                temp_local_correlation_matrix = temp_independence_test_metadata["local_correlation_matrix"]

                p_value += ((temp_mgc_statistic >= test_statistic) * (1/replication_factor))
                p_local_correlation_matrix += ((temp_local_correlation_matrix >=
                                                local_correlation_matrix) * (1/replication_factor))

            p_value_metadata = {"test_statistic": test_statistic,
                                "p_local_correlation_matrix": p_local_correlation_matrix,
                                "local_correlation_matrix": local_correlation_matrix,
                                "optimal_scale": independence_test_metadata["optimal_scale"]}
        elif self.get_name() == "kendall":
            p_value = kendalltau(matrix_X, matrix_Y)[1]
            p_value_metadata = {}
        elif self.get_name() == "spearman":
            p_value = spearmanr(matrix_X, matrix_Y)[1]
            p_value_metadata = {}
        elif self.get_name() == "pearson":
            p_value = pearsonr(matrix_X, matrix_Y)[1]
            p_value_metadata = {}
        else:
            # estimate the null by a permutation test
            test_stats_null = np.zeros(replication_factor)
            for rep in range(replication_factor):
                permuted_y = np.random.permutation(matrix_Y)
                test_stats_null[rep], _ = self.test_statistic(matrix_X=matrix_X, matrix_Y=permuted_y)
            # p-value is the probability of observing more extreme test statistic under the null
            p_value = np.where(test_stats_null >= test_statistic)[0].shape[0] / replication_factor
            p_value_metadata = {}

        # The results are not statistically significant
        if p_value > 0.05:
            warnings.warn("The p-value is greater than 0.05, implying that the results are not statistically significant.\n" +
                          "Use results such as test_statistic and optimal_scale, with caution!")

        self.p_value_ = p_value
        self.p_value_metadata_ = p_value_metadata
        return p_value, p_value_metadata
