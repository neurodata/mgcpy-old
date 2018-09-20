#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:54:53 2018

@author: spanda
"""

import numpy as np

def rVCorr(mat1, mat2, *option):
    
    try:
        option
    except NameError:
        option = 0
        
    sizeX, sizeY = np.size(mat1)
    
    mat1 = mat1 - np.matlib.repmat(np.mean(mat1, 1), sizeX, 1)
    mat2 = mat2 - np.matlib.repmat(np.mean(mat2, 1), sizeX, 1)
    
    covariance = np.transpose(mat1) * mat2
    varianceX = np.transpose(mat1) * mat1
    varianceY = np.transpose(mat2) * mat2
    
    option = np.min(np.abs(option), sizeY)
    
    if (option == 0):
        covariance = np.trace(covariance * np.transpose(covariance))
        correlation = covariance / np.sqrt(np.trace(varianceX * varianceX) * np.trace(varianceY, varianceY))
    else:
        covariance = np.sum(np.linalg.svd(covariance, option) ** 2)
        correlation = covariance / np.sqrt(np.sum(np.linalg.svd(varianceX, option) ** 2) * np.sum(np.linalg.svd(varianceY, option) ** 2))
    
    return correlation
    