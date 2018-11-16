# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:19:02 2018

@author: Ananya S
"""

import numpy as np
import pandas as pd
from dcorr import DCorr
from mgc import MGC
from rv_corr import RVCorr
from hhg import HHG
from scipy.spatial.distance import pdist, squareform

class Two_Sample():
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix,ind_test='dcorr'):
        self.ind_test = ind_test
        trans_X,trans_Y = Transform_Matrices(data_matrix_X,data_matrix_Y)
        independence(trans_X,trans_Y)

    def independence(trans_X,trans_Y):
        if self.ind_test=='dcorr':
            return DCorr(trans_X,trans_Y,compute_distance_matrix)
        if self.ind_test=='mgc':
            return MGC(trans_X,trans_Y,compute_distance_matrix)
        if self.ind_test=='rv_corr':
            return RVCorr(trans_X,trans_Y,compute_distance_matrix)
        if self.ind_test=='hhg':
            return HHG(trans_X,trans_Y,compute_distance_matrix)
    
    def compute_distance_matrix(data_matrix_X, data_matrix_Y):
        # obtain the pairwise distance matrix for X and Y
        dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
        dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
        return (dist_mtx_X, dist_mtx_Y)

    def Transform_Matrices(A,B):
        U=A.tolist()
        V=B.tolist()
        if isinstance(U[0],list):
            col=len(U[0])+len(V[0])
            row=len(U)
        else:
            col=len(U)+len(V)
            row=1
                
        data=[[0 for t1 in range (col)] for t2 in range (row)]
        num=[0 for t1 in range (col)]
                
        if isinstance(U[0],list):
            for n1 in range (row):
                for n2 in range (len(U[0])):
                    data[n1][n2]=U[n1][n2]
                    num[n2]=0
            for n3 in range (row):
                for n4 in range (len(V[0])):
                    data[n3][n4+len(U[0])]=V[n3][n4]
                    num[n4+len(U[0])]=1
            else:
                for n1 in range (row):
                    for n2 in range (len(U)):
                        data[n1][n2]=U[n2]
                        num[n2]=0
                for n3 in range (row):
                    for n4 in range (len(V)):
                        data[n3][n4+len(U)]=V[n4]
                        num[n4+len(U)]=1
                        
        x=np.asarray(data)
        y=np.asarray(num)
        return x,y
                                                    