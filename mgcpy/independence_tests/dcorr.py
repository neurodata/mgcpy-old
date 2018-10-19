import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t
import math
from mgcpy.utils.dist_transform import dist_transform
from mgcpy.independence_tests.abstract_class import IndependenceTest


class DCorr(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, corr_type='mcorr', is_distance_mtx=False):
        '''
        :param data_matrix_X: [n*p], n: number of examples, p: dimension of each example
        :type: 2D numpy.array

        :param data_matrix_Y: [n*q], n: number of examples, q: dimension of each example
        :type: 2D numpy array

        :param corr_type: the type of global correlation to use, can be 'dcorr', 'mcorr', 'mantel'
        :type: str

        :param is_distance_mtx: whether the given data matrices are already distance matrices
        :type: boolean
        '''
        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.corr_type = corr_type
        self.is_distance_mtx = is_distance_mtx

    def test_statistic(self, data_matrix_X=None, data_matrix_Y=None):
        '''
        Compute the correlation between data_matrix_X and data_matrix_Y using dcorr/mcorr/mantel

        Procedure: compute two distance matrices, each n*n using pdist and squareform
        then perform distance transformation using dist_transform()
        calculate correlation by computing all global covariance and variance using global_cov(A, B)

        :param data_matrix_X: optional data matrix
        :type: 2D numpy array

        :param data_matrix_Y: optional data matrix
        :type: 2D numpy array

        :return: the value of the correlation test statistic
        :rtype: float
        '''
        # if no data matrix is given, use the data matrices given at initialization
        if data_matrix_X is None and data_matrix_Y is None:
            data_matrix_X = self.data_matrix_X
            data_matrix_Y = self.data_matrix_Y

        # if the matrices given are already distance matrices, skip computing distance matrices
        if self.is_distance_mtx:
            dist_mtx_X = data_matrix_X
            dist_mtx_Y = data_matrix_Y
        else:
            dist_mtx_X, dist_mtx_Y = self.compute_distance_matrix(data_matrix_X=data_matrix_X, data_matrix_Y=data_matrix_Y)

        # perform distance transformation
        transformed_dist_mtx_X, transformed_dist_mtx_Y = dist_transform(dist_mtx_X, dist_mtx_Y, self.corr_type)
        # transformed_dist_mtx need not be symmetric
        covariance = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_Y))
        variance_X = self.compute_global_covariance(transformed_dist_mtx_X, np.transpose(transformed_dist_mtx_X))
        variance_Y = self.compute_global_covariance(transformed_dist_mtx_Y, np.transpose(transformed_dist_mtx_Y))

        # check the case when one of the dataset has zero variance
        if variance_X <= 0 or variance_Y <= 0:
            correlation = 0
        else:
            correlation = covariance/np.real(np.sqrt(variance_X*variance_Y))

        # use absolute value for mantel coefficients
        if self.corr_type == 'mantel':
            return np.abs(correlation)

        return correlation

    def compute_global_covariance(self, dist_mtx_X, dist_mtx_Y):
        '''
        Compute the global covariance using distance matrix A and B

        :param A, B: n*n distance matrix
        :return: float representing the covariance/variance
        '''
        return np.sum(np.multiply(dist_mtx_X, dist_mtx_Y))

    def mcorr_T(self, data_matrix_X, data_matrix_Y):
        '''
        Helper function: compute the t-test statistic for mcorr

        :param data_matrix_X: data matrix X
        :type: 2D numpy array

        :param data_matrix_Y: data matrix Y
        :type: 2D numpy array

        :return: test statistic of t-test for mcorr
        :rtype: np.float
        '''
        test_stat = self.test_statistic(data_matrix_X=data_matrix_X, data_matrix_Y=data_matrix_Y)

        n = data_matrix_X.shape[0]
        if n < 4:
            print('Not enough samples')
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

    def p_value(self, repeats=1000):
        '''
        Compute the p-value
        if the correlation test is mcorr, p-value can be computed using a t test
        otherwise computed using permutation test

        :return: float representing the p-value
        '''
        # calculte the test statistic with the given data
        test_stat = self.test_statistic()
        if self.corr_type == 'mcorr':
            '''
            for the unbiased centering scheme used to compute mcorr test statistic
            we can use a t-test to compute the p-value
            notation follows from: Székely, Gábor J., and Maria L. Rizzo.
            "The distance correlation t-test of independence in high dimension."
            Journal of Multivariate Analysis 117 (2013): 193-213.
            '''
            T, df = self.mcorr_T(data_matrix_X=self.data_matrix_X, data_matrix_Y=self.data_matrix_Y)
            # p-value is the probability of obtaining values more extreme than the test statistic
            # under the null
            if T < 0:
                return t.cdf(T, df=df)
            else:
                return 1 - t.cdf(T, df=df)
        else:
            # estimate the null by a permutation test
            test_stats_null = np.zeros(repeats)
            for rep in range(repeats):
                permuted_y = np.random.permutation(self.data_matrix_Y)
                test_stats_null[rep] = self.test_statistic(data_matrix_X=self.data_matrix_X, data_matrix_Y=permuted_y)
            # p-value is the probability of observing more extreme test statistic under the null
            return np.where(test_stats_null >= test_stat)[0].shape[0] / repeats
