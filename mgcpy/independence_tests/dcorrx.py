from mgcpy.independence_tests.ts_abstract_class import TimeSeriesIndependenceTest
from mgcpy.independence_tests.dcorr import DCorr

class DCorrX(TimeSeriesIndependenceTest):

    def __init__(self, compute_distance_matrix=None, which_test = 'unbiased', max_lag = 0):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: ``FunctionType`` or ``callable()``

        :param which_test: Whether to use 'unbiased' or 'biased' DCorrX.
        :type which_test: string

        :param max_lag: Furthest lag to check for dependence. Defaults to log(n).
        :type max_lag: integer
        '''
        if which_test not in ['unbiased', 'biased']:
            raise ValueError('which_test must be unbiased or biased.')
        super().__init__(DCorr(compute_distance_matrix = compute_distance_matrix,
                               which_test = which_test),
                         which_test = which_test,
                         compute_distance_matrix = compute_distance_matrix,
                         max_lag = max_lag)

    def test_statistic(self, matrix_X, matrix_Y):
        """"
        Test statistic for DCorrX between two time series.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
        :type matrix_Y: 2D numpy.array

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample test statistic (between [0,M])
            - :test_statistic_metadata: a ``dict`` of metadata with the following keys:
                    - :optimal_lag: the lag of maximal dependence
                    - :dependence_by_lag: the test_statistic restricted to each lag.
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorrx import DCorrX
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> dcorrx = DCorrX(max_lag = 3)
        >>> dcorrx_statistic, _ = dcorrx.test_statistic(X, Y)
        """
        return super().test_statistic(matrix_X, matrix_Y)

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000, is_fast = False, block_size = None, subsample_size = -1):
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

        :param is_fast: whether to use Fast DCorrX.
        :type is_fast: boolean

        :param block_size: Block size for block permutation procedure. Default sqrt(n).
        :type block_size: integer

        :return: returns a list of two items, that contains:

            - :p_value: P-value of DCorrX
            - :metadata: a ``dict`` of metadata with the following keys:
                    - :null_distribution: numpy array representing distribution of test statistic under null.
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorrx import DCorrX
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> dcorrx = DCorrX()
        >>> p_value, metadata = dcorrx.p_value(X, Y, replication_factor = 100)
        """
        return super().p_value(matrix_X, 
                            matrix_Y,
                            replication_factor = replication_factor,
                            is_fast = is_fast, 
                            block_size = block_size,
                            subsample_size = subsample_size)
