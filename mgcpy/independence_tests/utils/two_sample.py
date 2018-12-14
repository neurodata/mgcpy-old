import numpy as np
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.hhg import HHG
from mgcpy.independence_tests.kendall_spearman import KendallSpearman
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.utils.transform_matrices import \
    transform_matrices


class TwoSample:
    def __init__(self, independence_test_type='dcorr_unbiased'):
        '''
        :param independence_test_type: the independence test to call; options include dcorr_unbiased, dcorr_biased, mantel, mgc, rv_corr, cca, pearson, hhg, kendall, spearman
        :type: str
        '''
        self.independence_test_type = independence_test_type

    def transform(self, matrix_X, matrix_Y):
        '''
        Transform two data matrices into one concatenated matrix and one label matrix

        Procedure: Concatenate the two data matrices into one matrix.
        In the label matrix, assign each element from matrix_X 0 and each element from matrix_Y 1.

        :param matrix_X: data matrix
        :type: numpy array
        :param matrix_Y: data matrix
        :type: numpy array

        :return: two data matrices
        :rtype: numpy arrays
        '''
        x, y = transform_matrices(matrix_X, matrix_Y)
        y = y[:, np.newaxis]
        x = x.T
        return x, y

    def test(self, matrix_X, matrix_Y):
        '''
        Compute the correlation between matrix_X and matrix_Y using indicated test

        :param matrix_X: data matrix
        :type: 2D numpy array
        :param matrix_Y: data matrix
        :type: 2D numpy array

        :return: returns a list of two items, that contains:
            - :test_statistic: the test statistic computed using the respective independence test
            - :p-value
        :rtype: float, float
        '''
        X, Y = self.transform(matrix_X, matrix_Y)
        if self.independence_test_type == 'dcorr_unbiased':
            dcorr = DCorr()
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.independence_test_type == 'dcorr_biased':
            dcorr = DCorr(which_test='biased')
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.independence_test_type == 'mantel':
            dcorr = DCorr(which_test='mantel')
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.independence_test_type == 'mgc':
            mgc = MGC()
            return mgc.test_statistic(X, Y), mgc.p_value(X, Y)
        if self.independence_test_type == 'rv_corr':
            rv_corr = RVCorr()
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.independence_test_type == 'cca':
            rv_corr = RVCorr(which_test='cca')
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.independence_test_type == 'pearson':
            rv_corr = RVCorr(which_test='pearson')
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.independence_test_type == 'hhg':
            hhg = HHG()
            return hhg.test_statistic(X, Y), hhg.p_value(X, Y)
        if self.independence_test_type == 'kendall':
            ks = KendallSpearman(which_test='kendall')
            return ks.test_statistic(X, Y), ks.p_value(X, Y)
        if self.independence_test_type == 'spearman':
            ks = KendallSpearman(which_test='spearman')
            return ks.test_statistic(X, Y), ks.p_value(X, Y)
