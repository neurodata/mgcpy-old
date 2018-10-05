from abc import ABC, abstractmethod


class IndependenceTest(ABC):
    """
    IndependenceTest abstract class

    Specifies the generic interface that must be implemented by
    all the independence tests in the mgcpy package.

    :param data_matrix_X: a [n*p] data matrix, a square matrix with n samples in p dimensions
    :type data_matrix_X: 2D `numpy.array`

    :param data_matrix_Y: a [n*q] data matrix, a square matrix with n samples in q dimensions
    :type data_matrix_Y: 2D `numpy.array`

    :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
    :type compute_distance_matrix: FunctionType or callable()
    """

    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix):
        self.data_matrix_X = data_matrix_X
        self.data_matrix_Y = data_matrix_Y
        self.compute_distance_matrix = compute_distance_matrix
        super().__init__()

    @abstractmethod
    def test_statistic(self):
        """
        Abstract method to compute the test statistic,
        given `self.data_matrix_X` and `self.data_matrix_Y`.

        :return: returns a list of two items, that contains:
            - :test_statistic: the test statistic computed using the respective independence test
            - :independence_test_metadata: (optional) metadata other than the test_statistic,
                                           that the independence tests computes in the process
        :rtype: float, dict
        """
        pass
