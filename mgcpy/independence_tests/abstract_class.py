from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, squareform


class IndependenceTest(ABC):
    """
    IndependenceTest abstract class

    Specifies the generic interface that must be implemented by
    all the independence tests in the mgcpy package.

    :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
    :type compute_distance_matrix: FunctionType or callable()
    """

    def __init__(self, compute_distance_matrix=None):
        self.test_statistic = None
        self.test_statistic_metadata = None
        self.p_value = None
        self.p_value_metadata = None

        if not compute_distance_matrix:
            def EUCLIDEAN_DISTANCE(x): return squareform(pdist(x, metric='euclidean'))
            compute_distance_matrix = EUCLIDEAN_DISTANCE
        self.compute_distance_matrix = compute_distance_matrix

        super().__init__()

    @abstractmethod
    def get_name(self):
        '''
        :return: the name of the independence test
        :rtype: string
        '''
        pass

    @abstractmethod
    def test_statistic(self, data_matrix_X, data_matrix_Y):
        """
        Abstract method to compute the test statistic given two data matrices

        :param data_matrix_X: a [n*p] data matrix, a square matrix with n samples in p dimensions
        :type data_matrix_X: 2D `numpy.array`

        :param data_matrix_Y: a [n*q] data matrix, a square matrix with n samples in q dimensions
        :type data_matrix_Y: 2D `numpy.array`

        :return: returns a list of two items, that contains:
            - :test_statistic: the test statistic computed using the respective independence test
            - :test_statistic_metadata: (optional) metadata other than the test_statistic,
                                        that the independence tests computes in the process
        :rtype: float, dict
        """
        pass

    @abstractmethod
    def p_value(self, data_matrix_X, data_matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using MGC and permutation test.

        :param data_matrix_X: a [n*p] data matrix, a square matrix with n samples in p dimensions
        :type data_matrix_X: 2D `numpy.array`

        :param data_matrix_Y: a [n*q] data matrix, a square matrix with n samples in q dimensions
        :type data_matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to 1000.
        :type replication_factor: int

        :return: returns a list of two items, that contains:
            - :p_value: P-value of MGC
            - :p_value_metadata: (optional) a ``dict`` of metadata other than the p_value,
                                 that the independence tests computes in the process
        """
        pass
