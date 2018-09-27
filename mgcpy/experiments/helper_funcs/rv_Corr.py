#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial.distance import squareform, pdist


def rVCorr(mat1, mat2, option=0):  
    """
    Main function that calculates all the local correlation coefficients.
    
    :param mat1: a n-dimensional data matrix
    :param mat2: a n-dimensional data matrix
    :param option: a number that specifies which global correlation to use, 
                   including 'mcor','dcor','mantel'.
                   
    :return: The local correlation ''corr'' and local covaraince ''covar'' of
             ''mat1'' and ''mat2''
    """
    mat1[np.isnan(mat1)] = 0
    mat2[np.isnan(mat2)] = 0
    if (np.allclose(mat1.T, mat1) == False):
        mat1 = squareform(pdist(mat1));
    if (np.allclose(mat2.T, mat2) == False):
        mat2=squareform(pdist(mat2));
    
    sizeX = mat1.shape[0]
    sizeY = mat1.shape[1]
    
    mat1 = mat1 - mb.repmat(np.mean(mat1, 1), sizeX, 1)
    mat2 = mat2 - mb.repmat(np.mean(mat2, 1), sizeX, 1)
    
    covar = mat1.T * mat2
    varX = mat1.T * mat1
    varY = mat2.T * mat2
    
    option = np.minimum(np.abs(option), sizeY)
    if (option == 0):
        covar = np.trace(covar * covar.T)
        corr = np.divide(covar, np.sqrt(np.trace(varX * varX)
                                * np.trace(varY * varY)))
    else:
        covar = np.sum(np.power(svds(covar, option), 2))
        corr = np.divide(covar, np.sqrt(np.sum(np.power(svds(varX, option), 2))
                                * np.sum(np.power(svds(varY, option), 2))))
    
    return corr, covar