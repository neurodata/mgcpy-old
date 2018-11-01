import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from scipy.spatial import distance_matrix


class HHG(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix):
        """
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
        """
        IndependenceTest.__init__(
            self, data_matrix_X, data_matrix_Y, compute_distance_matrix)

    def test_statistic(self, data_matrix_X=None, data_matrix_Y=None):
        """
        Computes the HHG correlation measure between two datasets.

        :param data_matrix_X: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array

        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array

        :return: the sample test statistic

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.hhg import HHG

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> hhg = HHG(X, Y, None)
        >>> hhg_stat = hhg.test_statistic()
        """
        if data_matrix_X is None:
            data_matrix_X = self.data_matrix_X
        if data_matrix_Y is None:
            data_matrix_Y = self.data_matrix_Y

        row_X, columns_X = data_matrix_X.shape[0], data_matrix_X.shape[1]
        row_Y, columns_Y = data_matrix_Y.shape[0], data_matrix_Y.shape[1]

        # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
        if row_X != columns_X or sum(data_matrix_X.diagonal()**2) > 0:
            dist_mtx_X = distance_matrix(data_matrix_X, data_matrix_X)
        else:
            dist_mtx_X = data_matrix_X
        if row_Y != columns_Y or sum(data_matrix_Y.diagonal()**2) > 0:
            dist_mtx_Y = distance_matrix(data_matrix_Y, data_matrix_Y)
        else:
            dist_mtx_Y = data_matrix_Y

        n = dist_mtx_X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    tmp1 = dist_mtx_X[i, :] <= dist_mtx_X[i, j]
                    tmp2 = dist_mtx_Y[i, :] <= dist_mtx_Y[i, j]
                    t11 = np.sum(tmp1 * tmp2) - 2
                    t12 = np.sum(tmp1 * (1-tmp2))
                    t21 = np.sum((1-tmp1) * tmp2)
                    t22 = np.sum((1-tmp1) * (1-tmp2))
                    denom = (t11+t12) * (t21+t22) * (t11+t21) * (t12+t22)
                    if denom > 0:
                        S[i, j] = (n-2) * \
                            np.power((t12*t21 - t11*t22), 2) / denom
        corr = np.sum(S)

        return corr

    def p_value(self, replication_factor=1000):
        """
        Tests independence between two datasets using HHG and permutation test.

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: P-value of HHG

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.hhg import HHG

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> hhg = HHG(X, Y, None)
        >>> p_value = hhg.p_value(replication_factor = 100)
        """
        test_stat = self.test_statistic()
        # estimate the null by a permutation test
        test_stats_null = np.zeros(replication_factor)
        for rep in range(replication_factor):
            permuted_y = np.random.permutation(self.data_matrix_Y)
            test_stats_null[rep] = self.test_statistic(data_matrix_X=self.data_matrix_X, data_matrix_Y=permuted_y)

        # p-value is the probability of observing more extreme test statistic under the null
        return np.where(test_stats_null >= test_stat)[0].shape[0] / replication_factor
