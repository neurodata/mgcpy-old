# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:42:35 2018

@author: sunda
"""


from mdmrpy import *

#!python
#!/usr/bin/env python
import scipy.io as scpio
import scipy as scp
import numpy as np
#from scipy.io import loadmat
#x = scpio.loadmat('x.mat')['x']


csv1 = np.genfromtxt('X_mdmr.csv', delimiter=",")
X = csv1

csv1 = np.genfromtxt('Y_mdmr.csv', delimiter=",")
Y = csv1

D = scp.spatial.distance.pdist(Y, 'cityblock')
D = scp.spatial.distance.squareform(D)
a = D.shape[0]**2
D = D.reshape((a,1))
#print(D.shape)
#print(X.shape)

#####################################################
#columns = 1
#permutations = 100
#print ("Column =", columns)

results = mdmr(D,X)
for i in range(0,results.shape[0]):
    print("Column =", int(results[i,0]))
    print("F_Perm =", results[i,1])
    print("P-Value =", results[i,2])
