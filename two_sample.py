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

class Two_Sample:
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix,ind_test='dcorr'):
        self.ind_test = ind_test
        self.X=data_matrix_X
        self.Y=data_matrix_Y
        self.trans_X,self.trans_Y = self.Transform_Matrices(data_matrix_X,data_matrix_Y)
        self.independence()
        '''
        :param data_matrix_X: data matrix
        :type: numpy array
        :param data_matrix_Y: data matrix
        :type: numpy array
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type: FunctionType or callable()
        :param ind_test: the independence test to call
        :type: str
        '''
    def compute_distance_matrix(self):
        # obtain the pairwise distance matrix for X and Y
        dist_mtx_X = squareform(pdist(self.X, metric='euclidean'))
        dist_mtx_Y = squareform(pdist(self.Y, metric='euclidean'))
        return (dist_mtx_X, dist_mtx_Y)
        '''
        :param self: object
        :type: Two Sample Object
        :return: distance matrices for X and Y
        :rtype: matrices
        '''

    def independence(self):
        #choose independence test to run
        if self.ind_test=='dcorr':
            return DCorr(self.trans_X,self.trans_Y,self.compute_distance_matrix)
        if self.ind_test=='mgc':
            return MGC(self.trans_X,self.trans_Y,self.compute_distance_matrix)
        if self.ind_test=='rv_corr':
            return RVCorr(self.trans_X,self.trans_Y,self.compute_distance_matrix)
        if self.ind_test=='hhg':
            return HHG(self.trans_X,self.trans_Y,self.compute_distance_matrix)

    def Transform_Matrices(self,A,B):
        #transform two data matrices into one concatenated matrix and one label matrix
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
                        
        self.x=np.asarray(data)
        self.y=np.asarray(num)
        return self.x,self.y
                                                    