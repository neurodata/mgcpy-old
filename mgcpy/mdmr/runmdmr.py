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
columns = 1
#permutations = 100
print ("Column =", columns)

[a,b] = mdmr(D,X,columns, permutations)
print ("Fperm Statistic =", a, "P-value =", b)
#####################################################
columns = 2
print ("Column =", columns)

[a,b] = mdmr(D,X,columns, permutations)
print ("Fperm Statistic =", a, "P-value =", b)
######################################################
columns = 3
print ("Column =", columns)

[a,b] = mdmr(D,X,columns, permutations)
print ("Fperm Statistic =", a, "P-value =", b)