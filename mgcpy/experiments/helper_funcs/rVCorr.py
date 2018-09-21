#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:54:53 2018

@author: spanda
"""

""" Function calculates local correlation coefficients"""

import numpy as np
from numpy import matlib as mb
from scipy.sparse.linalg import svds

def rVCorr(mat1, mat2, option):  
       
    sizeX, sizeY = np.size(mat1, 0), np.size(mat1, 1)
    
    mat1 = mat1 - mb.repmat(np.mean(mat1, 1), sizeX, 1)
    mat2 = mat2 - mb.repmat(np.mean(mat2, 1), sizeX, 1)
    
    covariance = np.transpose(mat1) * mat2
    varianceX = np.transpose(mat1) * mat1
    varianceY = np.transpose(mat2) * mat2
    
    option = np.minimum(np.abs(option), sizeY)
    
    if (option == 0):
        covariance = np.trace(covariance * np.transpose(covariance))
        correlation = np.divide(covariance, np.sqrt(np.trace(varianceX * varianceX) * np.trace(varianceY * varianceY)))
    else:
        covariance = np.sum(np.power(svds(covariance, option), 2))
        correlation = np.divide(covariance, np.sqrt(np.sum(np.power(svds(varianceX, option), 2)) * np.sum(np.power(svds(varianceY, option), 2))))
    
    return correlation

A = np.array([[0, 23, 56, 90, 5, 63, 49], 
     [23, 0, 80, 15, 95, 4, 43], 
     [56, 80, 0, 94, 27, 41, 90], 
     [90, 15, 94, 0, 95, 95, 89], 
     [5, 95, 27, 95, 0, 35, 37], 
     [63, 4, 41, 95, 35, 0, 31], 
     [49, 43, 90, 89, 37, 11, 0]])

B = np.array([[0, 18, 5, 3, 6, 75, 72], 
     [18, 0, 75, 46, 50, 6, 41], 
     [5, 75, 0, 35, 31, 44, 92], 
     [3, 46, 35, 0, 87, 62, 55], 
     [6, 50, 31, 87, 0, 57, 55], 
     [75, 6, 44, 62, 57, 0, 45], 
     [72, 41, 92, 55, 55, 45, 0]])

rVCorr(A, A, 0)
rVCorr(A, B, 1)