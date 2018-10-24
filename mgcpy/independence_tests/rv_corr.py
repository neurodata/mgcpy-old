#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy import matlib as mb
from scipy.sparse.linalg import svds

from mgcpy.independence_tests.abstract_class import IndependenceTest


class RVCorr(IndependenceTest):
    """
    Calculates the pearsons/Cca/rv correlation statistic.
    
    :param data_matrix_X: an input distance matrix
    :param data_matrix_Y: an input distance matrix
    :param compute_distance_matrix: a function to compute the pairwise distance
                                    matrix
    :param option: a number that specifies which global correlation to use, 
                   including 'mcor','dcor','mantel', defaults to 0
    """
    
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix, 
                 option=0, is_distance_mtx=False):
        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, 
                                  compute_distance_matrix)
        self.option = option
        self.is_distance_mtx = is_distance_mtx
        
    def test_statistic(self):
        """
        Calculates all the local correlation coefficients.
    
        :return: The local correlation ``corr`` and local covaraince ``covar`` 
                 of ``mat1`` and ``mat2``
        """
        
        # if no data matrix is given, use the data matrices given at initialization
        if self.data_matrix_X is None and self.data_matrix_Y is None:
            self.data_matrix_X = self.data_matrix_X
            self.data_matrix_Y = self.data_matrix_Y

        # if the matrices given are already distance matrices, skip computing distance matrices
        if self.is_distance_mtx:
            dist_mtx_X = self.data_matrix_X
            dist_mtx_Y = self.data_matrix_Y
        else:
            dist_mtx_X, dist_mtx_Y = \
            self.compute_distance_matrix(data_matrix_X=self.data_matrix_X, 
                                         data_matrix_Y=self.data_matrix_Y)
        
        mat1 = dist_mtx_X - mb.repmat(np.mean(dist_mtx_X, axis=0), 
                                              dist_mtx_X.shape[0], 1)
        mat2 = dist_mtx_Y - mb.repmat(np.mean(dist_mtx_Y, axis=0), 
                                              dist_mtx_Y.shape[0], 1)
        
        covar = np.matmul(a=mat1.T, b=mat2)
        varX = np.matmul(a=mat1.T, b=mat1)
        varY = np.matmul(a=mat2.T, b=mat2)
        
        self.option = np.minimum(np.abs(self.option), mat1.shape[1])
        if (self.option == 0):
            covar = np.trace(np.matmul(covar, covar.T))
            corr = np.divide(covar, np.sqrt(np.trace(np.matmul(varX, varX))
                                            * np.trace(np.matmul(varY, varY))))
        else:
            covar = np.sum(np.power(svds(covar, self.option)[1], 2))
            corr = np.divide(covar, np.sqrt(np.sum(np.power(svds(varX, self.option)[1], 2)) 
                * np.sum(np.power(svds(varY, self.option)[1], 2))))
        
        return [corr, covar]