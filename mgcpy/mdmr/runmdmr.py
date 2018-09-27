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
x = scpio.loadmat('x.mat')['x']
D = scp.spatial.distance.pdist(x, 'euclidean')
D = scp.spatial.distance.squareform(D)
D = D.reshape((10000,1))
print(D.shape)
print(x.shape)
columns = 100
mdmr(D,x,columns, 5)