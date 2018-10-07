import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t
from mgcpy.utils.dist_transform import dist_transform
from mgcpy.independence_tests.abstract_class import IndependenceTest


class DCorr(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, corr_type='mcorr'):
        '''
        :param data_matrix_X: [n*p], n: number of examples, p: dimension of each example
        :type: 2D numpy.array

        :param data_matrix_Y: [n*q], n: number of examples, q: dimension of each example
        :type: 2D numpy array

        :param corr_type: the type of global correlation to use, can be 'dcorr', 'mcorr', 'mantel'
        :type: str
        '''
        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
        self.corr_type = corr_type
        self.test_stat = None

    def test_statistic(self):
        '''
        Compute the correlation between data_matrix_X and data_matrix_Y using dcorr/mcorr/mantel

        Procedure: compute two distance matrices, each n*n using pdist and squareform
        then perform distance transformation using dist_transform()
        calculate correlation by computing all global covariance and variance using global_cov(A, B)

        :return: the value of the correlation test statistic
        :rtype: float
        '''

        dist_mtx_X, dist_mtx_Y = self.compute_distance_matrix(data_matrix_X=self.data_matrix_X, data_matrix_Y=self.data_matrix_Y)
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

        self.test_stat = correlation
        return self.test_stat

    '''
    def compute_distance_matrix(self):
        # obtain the pairwise distance matrix for X and Y
        dist_mtx_X = squareform(pdist(self.data_matrix_X, metric=self.metric))
        dist_mtx_Y = squareform(pdist(self.data_matrix_Y, metric=self.metric))
        return (dist_mtx_X, dist_mtx_Y)
    '''

    def compute_global_covariance(self, dist_mtx_X, dist_mtx_Y):
        '''
        Compute the global covariance using distance matrix A and B

        :param A, B: n*n distance matrix
        :return: float representing the covariance/variance
        '''
        return np.sum(np.multiply(dist_mtx_X, dist_mtx_Y))

    def p_value(self):
        '''
        Compute the p-value
        if the correlation test is mcorr, p-value can be computed using a t test
        otherwise computed using permutation test

        :return: float representing the p-value
        '''
        # calculte the test statistic if haven't done so
        if not self.test_stat:
            self.test_statistic()
        if self.corr_type == 'mcorr':
            n = self.data_matrix_X.shape[0]
            if n < 4:
                print('Not enough samples')
                return None
            '''
            for the unbiased centering scheme used to compute mcorr test statistic
            we can use a t-test to compute the p-value
            notation follows from: Székely, Gábor J., and Maria L. Rizzo.
            "The distance correlation t-test of independence in high dimension."
            Journal of Multivariate Analysis 117 (2013): 193-213.
            '''
            v = n*(n-3)/2
            # T converges in distribution to a t distribution under the null
            if self.test_stat == 1:
                '''
                if test statistic is 1, the t test statistic goes to inf
                '''
                T = np.inf
            else:
                T = np.sqrt(v-1) * self.test_stat / np.sqrt((1-np.square(self.test_stat)))
            # p-value is the probability of obtaining values more extreme than the test statistic
            # under the null
            if T < 0:
                return t.cdf(T, df=v-1)
            else:
                return 1 - t.cdf(T, df=v-1)
        else:
            # permutation test
            pass

    def power(self):
        pass
