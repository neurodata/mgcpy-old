# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:19:02 2018

@author: Ananya S
"""

import numpy as np
import pandas as pd
from transform_matrices import Transform_Matrices
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr
from mgcpy.independence_tests.hhg import HHG
from mgcpy.independence_tests.kendall_spearman import KendallSpearman
from scipy.spatial.distance import pdist, squareform


class Two_Sample:
    def __init__(self, ind_test='dcorr_unbiased'):
        '''
        :param data_matrix_X: data matrix
        :type: numpy array
        :param data_matrix_Y: data matrix
        :type: numpy array
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type: FunctionType or callable()
        :param ind_test: the independence test to call; options include dcorr_unbiased, dcorr_biased, mantel, mgc, rv_corr, cca, pearson, hhg, kendall, spearman
        :type: str
        '''
        self.ind_test = ind_test

    def transform(self, data_matrix_X, data_matrix_Y):
        x, y = Transform_Matrices(data_matrix_X, data_matrix_Y)
        y = y[:, np.newaxis]
        x = x.T
        return x, y

    def independence(self, data_matrix_X, data_matrix_Y):
        X, Y = self.transform(data_matrix_X, data_matrix_Y)
        if self.ind_test == 'dcorr_unbiased':
            dcorr = DCorr()
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.ind_test == 'dcorr_biased':
            dcorr = DCorr(which_test='biased')
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.ind_test == 'mantel':
            dcorr = DCorr(which_test='mantel')
            return dcorr.test_statistic(X, Y), dcorr.p_value(X, Y)
        if self.ind_test == 'mgc':
            mgc = MGC()
            return mgc.test_statistic(X, Y), mgc.p_value(X, Y)
        if self.ind_test == 'rv_corr':
            rv_corr = RVCorr()
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.ind_test == 'cca':
            rv_corr = RVCorr(which_test='cca')
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.ind_test == 'pearson':
            rv_corr = RVCorr(which_test='pearson')
            return rv_corr.test_statistic(X, Y), rv_corr.p_value(X, Y)
        if self.ind_test == 'hhg':
            hhg = HHG()
            return hhg.test_statistic(X, Y), hhg.p_value(X, Y)
        if self.ind_test == 'kendall':
            ks = KendallSpearman(which_test='kendall')
            return ks.test_statistic(X, Y), ks.p_value(X, Y)
        if self.ind_test == 'spearman':
            ks = KendallSpearman(which_test='spearman')
            return ks.test_statistic(X, Y), ks.p_value(X, Y)
