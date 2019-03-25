from mgcpy.independence_tests.abstract_class import IndependenceTest
from statsmodels.multivariate.manova import MANOVA


class Manova(IndependenceTest):
    def __init__(self, compute_distance_matrix=None):
        """
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()
        """
        IndependenceTest.__init__(self, compute_distance_matrix)
        self.which_test = "manova"

    def test_statistic(self, matrix_X, matrix_Y):
        """
        Computes the Manova test statistic between two datasets.
        - uses statsmodels.multivariate.manova's implementation

        :param matrix_X: a [n*p] data matrix, a matrix with n samples in p dimensions, where p >= 2
        :type matrix_X: 2D numpy.array

        :param matrix_Y: a [n*q] data matrix, a matrix with n samples in q dimensions
        :type matrix_Y: 2D numpy.array

        :return: returns a list of two items, that contains:

            - :test_statistic: the manova test statistic
            - :test_statistic_metadata: (optional) a ``dict`` of metadata that the
                                        independence tests computes in the process
        :rtype: float, dict

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.manova import Manova

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 2)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312]).reshape(-1, 1)
        >>> manova = Manova()
        >>> manova_stat = manova.test_statistic(X, Y)
        """
        assert matrix_X.shape[0] == matrix_Y.shape[0], "Matrices X and Y need to be of dimensions [n, p] and [n, q], respectively, where p can be equal to q"

        # use Pillai's trace to compute MANOVA
        self.test_statistic_ = MANOVA(matrix_X, matrix_Y).mv_test().results['x0']['stat'].values[1, 0]

        self.test_statistic_metadata_ = {}

        return self.test_statistic_, self.test_statistic_metadata_

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using the independence test.

        :param matrix_X: a [n*p] data matrix, a matrix with n samples in p dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: a [n*q] data matrix, a matrix with n samples in q dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: returns a list of two items, that contains:

            - :p_value_: P-value
            - :p_value_metadata_: (optional) a ``dict`` of metadata other than the p_value,
                                 that the independence tests computes in the process
        :rtype: float, dict

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.manova import Manova

        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                      0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 2)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312]).reshape(-1, 1)
        >>> manova = Manova()
        >>> manova_stat = manova.p_value(X, Y)
        """
        return super(Manova, self).p_value(matrix_X, matrix_Y)
