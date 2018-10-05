#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial import distance_matrix


def rv_corr(mat1_data, mat2_data, option=0):
    """
    Main function that calculates all the local correlation coefficients.

    :param mat1: a n-dimensional data matrix
    :param mat2: a n-dimensional data matrix
    :param option: a number that specifies which global correlation to use,
                   including 'mcor','dcor','mantel'.

    :return: The local correlation ''corr'' and local covaraince ''covar'' of
             ''mat1'' and ''mat2''
    """
    mat1_data[np.isnan(mat1_data)] = 0
    mat2_data[np.isnan(mat2_data)] = 0

    if (mat1_data.shape[0] != mat1_data.shape[1]) \
            or (np.allclose(mat1_data.T, mat1_data)):
        mat1 = distance_matrix(mat1_data, mat1_data)

    if (mat2_data.shape[0] != mat2_data.shape[1]) \
            or (np.allclose(mat2_data.T, mat2_data)):
        mat2 = distance_matrix(mat2_data, mat2_data)

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
