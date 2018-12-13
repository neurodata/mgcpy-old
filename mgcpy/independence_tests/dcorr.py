import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.mgc.distance_transform import \
    transform_distance_matrix
from scipy.stats import t


class DCorr(IndependenceTest):

    def __init__(self, compute_distance_matrix=None, which_test='unbiased'):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()

        :param which_test: the type of global correlation to use, can be 'unbiased', 'biased' 'mantel'
        :type which_test: string
        '''
        IndependenceTest.__init__(self)
        if which_test not in ['unbiased', 'biased', 'mantel']:
            raise ValueError('which_test must be unbiased, biased, or mantel')
        self.which_test = which_test

    def test_statistic(self, matrix_X, matrix_Y):
        """
        Computes the distance correlation between two datasets.

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :return: returns a list of two items, that contains:

            - :test_statistic: the sample dcorr statistic within [-1, 1]
            - :independence_test_metadata: a ``dict`` of metadata with the following keys:
                    - :variance_X: the variance of the data matrix X
                    - :variance_Y: the variance of the data matrix Y
        :rtype: list

        **Example:**

        >>> import numpy as np
        >>> from mgcpy.independence_tests.dcorr import DCorr
        >>>
        >>> X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
        ...           0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
        >>> Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
        ...           1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)
        >>> dcorr = DCorr(which_test = 'unbiased')
        >>> dcorr_statistic, test_statistic_metadata = dcorr.test_statistic(X, Y)
        """

        # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
        if matrix_X.shape[0] != matrix_X.shape[1] or sum(matrix_X.diagonal()**2) > 0:
            matrix_X = self.compute_distance_matrix(matrix_X)
        if matrix_Y.shape[0] != matrix_Y.shape[1] or sum(matrix_Y.diagonal()**2) > 0:
            matrix_Y = self.compute_distance_matrix(matrix_Y)

        # perform distance transformation
        # transformed_dist_mtx_X, transformed_dist_mtx_Y = dist_transform(matrix_X, matrix_Y, self.which_test)

        transformed_distance_matrices = transform_distance_matrix(matrix_X, matrix_Y, base_global_correlation=self.which_test, is_ranked=False)
        transformed_dist_mtx_X = transformed_distance_matrices['centered_distance_matrix_A']
        transformed_dist_mtx_Y = transformed_distance_matrices['centered_distance_matrix_B']

        # transformed_dist_mtx need not be symmetric
        covariance = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_Y))
        variance_X = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_X))
        variance_Y = self.compute_global_covariance(transformed_dist_mtx_Y, np.transpose(transformed_dist_mtx_Y))

        # check the case when one of the dataset has zero variance
        if variance_X <= 0 or variance_Y <= 0:
            correlation = 0
        else:
            correlation = covariance/np.real(np.sqrt(variance_X*variance_Y))

        # store the variance of X, variance of Y and the covariace as metadata
        self.test_statistic_metadata_ = {'variance_X': variance_X, 'variance_Y': variance_Y, 'covariance': covariance}

        # use absolute value for mantel coefficients

        if self.which_test == 'mantel':
            self.test_statistic_ = np.abs(correlation)
        else:
            self.test_statistic_ = correlation

        return (self.test_statistic_, self.test_statistic_metadata_)

    def compute_global_covariance(self, dist_mtx_X, dist_mtx_Y):
        '''
        Helper function: Compute the global covariance using distance matrix A and B

        :param dist_mtx_X: a [n*n] distance matrix
        :type dist_mtx_X: 2D numpy.array

        :param dist_mtx_Y: a [n*n] distance matrix
        :type dist_mtx_Y: 2D numpy.array

        :return: the data covariance or variance based on the distance matrices
        :rtype: numpy.float
        '''
        return np.sum(np.multiply(dist_mtx_X, dist_mtx_Y))

    def unbiased_T(self, matrix_X, matrix_Y):
        '''
        Helper function: Compute the t-test statistic for unbiased dcorr

        :param matrix_X: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D numpy.array

        :param matrix_Y: is interpreted as either:

            - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for ``n`` samples OR
            - a ``[n*d]`` matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D numpy.array

        :return: test statistic of t-test for unbiased dcorr
        :rtype: numpy.float
        '''
        test_stat, _ = self.test_statistic(matrix_X=matrix_X, matrix_Y=matrix_Y)

        n = matrix_X.shape[0]
        if n < 4:
            raise ValueError('Not enough samples, number of samples must be greater than 3')
            return None
        v = n*(n-3)/2
        # T converges in distribution to a t distribution under the null
        if test_stat == 1:
            '''
            if test statistic is 1, the t test statistic goes to inf
            '''
            T = np.inf
        else:
            T = np.sqrt(v-1) * test_stat / np.sqrt((1-np.square(test_stat)))
        return (T, v-1)

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        '''
        Compute the p-value
        if the correlation test is unbiased, p-value can be computed using a t test
        otherwise computed using permutation test

        :return: p-value of distance correlation
        :rtype: numpy.float
        '''
        return super(DCorr, self).p_value(matrix_X, matrix_Y)
