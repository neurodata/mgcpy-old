import numpy as np
from mgcpy.independence_tests.abstract_class import IndependenceTest
from scipy.stats import t
from mgcpy.independence_tests.mgc.distance_transform import transform_distance_matrix


class DCorr(IndependenceTest):

    def __init__(self, compute_distance_matrix=None, which_test='unbiased'):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type: FunctionType or callable()

        :param which_test: the type of global correlation to use, can be 'unbiased', 'biased' 'mantel'
        :type: str
        '''
        IndependenceTest.__init__(self)
        if which_test not in ['unbiased', 'biased', 'mantel']:
            raise ValueError('which_test must be unbiased, biased, or mantel')
        self.which_test = which_test

    def test_statistic(self, matrix_X, matrix_Y):
        '''
        Compute the correlation between matrix_X and matrix_Y using biased/unbiased/mantel

        Procedure: compute two distance matrices, each n*n using pdist and squareform
        then perform distance transformation using dist_transform()
        calculate correlation by computing all global covariance and variance using global_cov(A, B)

        :param matrix_X: data matrix
        :type: 2D numpy array

        :param matrix_Y: data matrix
        :type: 2D numpy array

        :return: returns a list of two items, that contains:
            - :test_statistic: the test statistic computed using the respective independence test
            - :independence_test_metadata: (optional) metadata other than the test_statistic,
                                           that the independence tests computes in the process
        :rtype: float, dict
        '''

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
        Compute the global covariance using distance matrix A and B

        :param A, B: n*n distance matrix
        :return: float representing the covariance/variance
        '''
        return np.sum(np.multiply(dist_mtx_X, dist_mtx_Y))

    def unbiased_T(self, matrix_X, matrix_Y):
        '''
        Helper function: compute the t-test statistic for unbiased

        :param matrix_X: data matrix X
        :type: 2D numpy array

        :param matrix_Y: data matrix Y
        :type: 2D numpy array

        :return: test statistic of t-test for unbiased
        :rtype: np.float
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

        :return: float representing the p-value
        '''
        return super(DCorr, self).p_value(matrix_X, matrix_Y)
