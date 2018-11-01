import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from scipy.spatial import distance_matrix
from scipy.stats import kendalltau, spearmanr


class KendallSpearman(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, which_test='kendall'):
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

        :param which_test: specifies which test to use, including 'kendall' or 'spearman'
        :type which_test: str
        """
        IndependenceTest.__init__(
            self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.which_test = which_test

    def test_statistic(self, data_matrix_X, data_matrix_Y):
        """
        Computes the Spearman's rho or Kendall's tau measure between two datasets.
        - Implments scipy.stats's implementation for both

        :param data_matrix_X: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array

        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array

        :return: returns the sample test statistic

        **Example:**
        >>> import numpy as np
        >>> from mgcpy.independence_tests.kendall_spearman import KendallSpearman

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> rvcorr = RVCorr(X, Y, None)
        >>> rvcorr_stat, rvcorr_covar = rvcorr.test_statistic()
        """
        if data_matrix_X is None:
            data_matrix_X = self.data_matrix_X
        if data_matrix_Y is None:
            data_matrix_Y = self.data_matrix_Y

        if self.which_test == 'kendall':
            test_stat, p_value = kendalltau(data_matrix_X, data_matrix_Y)
        else:
            test_stat, p_value = spearmanr(data_matrix_X, data_matrix_Y)

        independence_test_metadata = {"p-value": p_value}

        return test_stat, independence_test_metadata
