# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:19:02 2018

@author: Ananya S
"""

import numpy as np
import pandas as pd
from scipy import spatial

def Transform_Matrices(U,V):
    sizeu=U.shape
    sizev=V.shape
    col=sizeu[0]+sizev[0]
    if len(sizeu)==1:
        row=1
    else:
        row=sizeu[1]
    data=[[0 for t1 in range (row)] for t2 in range (col)]
    num=[[0 for t1 in range (row)] for t2 in range (col)]
    for n in range (sizeu[0]):
        for n2 in range (row):
            data[n][n2]=U[n][n2]
            num[n][n2]=0
    x=np.asarray(data)
    y=np.asarray(num)
    print(x)
    