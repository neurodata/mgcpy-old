"""
     **Kendall and Spearman ndependence Test Module**
"""

from mgcpy.independence_tests.abstract_class import IndependenceTest
from scipy.stats import kendalltau, spearmanr


class KendallSpearman(IndependenceTest):
    def __init__(self, compute_distance_matrix=None, which_test='kendall'):
        """
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: ``FunctionType`` or ``callable()``

        :param which_test: specifies which test to use, including ``'kendall'`` or ``'spearman'``
        :type which_test: string
        """
        IndependenceTest.__init__(self, compute_distance_matrix)
        self.which_test = which_test

    def get_name(self):
        '''
        :return: the name of the independence test
        :rtype: string
        '''
        return self.which_test

    def test_statistic(self, matrix_X, matrix_Y):
        """
        Computes the Spearman's rho or Kendall's tau measure between two datasets.
        - Implments scipy.stats's implementation for both

        :param matrix_X: a ``[n*1]`` data matrix, a square matrix with ``n`` samples in ``1`` dimension
        :type matrix_X: 1D `numpy.array`

        :param matrix_Y: a ``[n*1]`` data matrix, a square matrix with ``n`` samples in ``1`` dimension
        :type matrix_Y: 1D `numpy.array`

        :return: returns a list of two items, that contains:
            - :test_stat_: test statistic
            - :test_statistic_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                         that the independence tests computes in the process
        :rtype: list

        **Example:**
        
        >>> import numpy as np
        >>> from mgcpy.independence_tests.kendall_spearman import KendallSpearman

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> kendall_spearman = KendallSpearman()
        >>> kendall_spearman_stat = kendall_spearman.test_statistic(X, Y)
        """
        assert matrix_X.shape[1] == 1, "Data matrix should be (n, 1) shape"
        assert matrix_Y.shape[1] == 1, "Data matrix should be (n, 1) shape"
        
        if self.which_test == 'kendall':            
            test_statistic_ = kendalltau(matrix_X, matrix_Y)[0]
        else:
            test_statistic_ = spearmanr(matrix_X, matrix_Y)[0]

        self.test_statistic_ = test_statistic_
        self.test_statistic_metadata_ = {}

        return test_statistic_, {}

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using the independence test.

        :param matrix_X: a ``[n*1]`` data matrix, a square matrix with ``n`` samples in ``1`` dimension
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a ``[n*1] data`` matrix, a square matrix with ``n`` samples in ``1`` dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: integer

        :return: returns a list of two items, that contains:
            - :p_value_: P-value
            - :p_value_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                 that the independence tests computes in the process
        :rtype: list

        **Example:**
        
        >>> import numpy as np
        >>> from mgcpy.independence_tests.kendall_spearman import KendallSpearman
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                      1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> kendall_spearman = KendallSpearman()
        >>> kendall_spearman_p_value = kendall_spearman.p_value(X, Y)
        """
        row_X, columns_X = matrix_X.shape[0], matrix_X.shape[1]
        row_Y, columns_Y = matrix_Y.shape[0], matrix_Y.shape[1]
        assert row_X == 1 or columns_X == 1, "Data matrix should be (n, 1) shape"
        assert row_Y == 1 or columns_Y == 1, "Data matrix should be (n, 1) shape"
        
        if self.which_test == 'kendall':
            p_value_ = kendalltau(matrix_X, matrix_Y)[1]
        else:
            p_value_ = spearmanr(matrix_X, matrix_Y)[1]

        self.p_value_ = p_value_
        self.p_value_metadata_ = {}

        return p_value_, {}