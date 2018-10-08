#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy import matlib as mb
from scipy.sparse.linalg import svds
from scipy.spatial import distance_matrix


def hsic(mat1, mat2):
    """
    Main function that calculates HSIC test statistic.

    :param mat1: a n-dimensional data matrix
    :param mat2: a n-dimensional data matrix

    :return: The calculated test statistic (HSIC)
    """
    mat1_hsic = hsic_kernel(mat1)
    mat2_hsic = hsic_kernel(mat2)
    m = mat1_hsic.shape[0]

    H = np.eye(m) - 1/m*np.ones(shape=(m,m))

    Kc = H * mat1_hsic * H
    test_stat = 1/(m**2) * np.sum(np.sum(np.multiply(Kc.T, mat2_hsic)))

    return test_stat


def hsic_kernel(x):
    """
    Generates kernel for HSIC test stat

    :param x:

    :return: HSIC kernel matrix for test stat
    """
    size_ = X.shape[0]
    if size_ > 100:
        xmed = x[1:100, :]
        size_ = 100
    else:
        xmed = x
    
    G = np.sum(np.multiply(xmed, xmed))