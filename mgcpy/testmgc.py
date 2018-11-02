# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:39:01 2018

@author: Ananya S
"""

import numpy as np
from scipy.sparse import coo_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mgcpy.independence_tests.mgc.mgc import MGC
import transform_matrices
from scipy.spatial.distance import pdist, squareform

edge_list_A_file = open("sub-NDARAE199TDD_acq-64dir_dwi_JHU_res-1x1x1_measure-spatial-ds.edgelist", "r")
edge_list_A = np.array([[int(t) for t in line.split()] for line in edge_list_A_file.readlines()])
node_list_A = sorted(list(set([int(i) for i, j, w in edge_list_A])))
adj_matrix_A = np.array(coo_matrix((edge_list_A[:, 2], (edge_list_A[:, 0]-1, edge_list_A[:, 1]-1)), shape=(len(node_list_A), len(node_list_A))).todense())
edge_list_A_file.close()

edge_list_B_file = open("sub-NDARAJ366ZFA_acq-64dir_dwi_JHU_res-1x1x1_measure-spatial-ds.edgelist", "r")
edge_list_B = np.array([[int(t) for t in line.split()] for line in edge_list_B_file.readlines()])
node_list_B = sorted(list(set([int(i) for i, j, w in edge_list_B])))
adj_matrix_B = np.array(coo_matrix((edge_list_B[:, 2], (edge_list_B[:, 0]-1, edge_list_B[:, 1]-1)), shape=(len(node_list_B), len(node_list_B))).todense())
edge_list_B_file.close()

A=adj_matrix_A.tolist()
B=adj_matrix_B.tolist()

X=transform_matrices.Transform_Matrices(A,B)[0]
Y=transform_matrices.Transform_Matrices(A,B)[1]

Y=Y[:,np.newaxis]
dist_mtx_X = squareform(pdist(X, metric='euclidean'))
dist_mtx_Y = squareform(pdist(Y, metric='euclidean'))
mgc = MGC(dist_mtx_X,dist_mtx_Y,None)
p_value, metadata = mgc.p_value()

# Define two rows for subplots
fig, (ax, cax) = plt.subplots(ncols=2, figsize=(9.45, 7.5),  gridspec_kw={"width_ratios":[1, 0.05]})
# Draw heatmap
# ax = sns.heatmap(metadata["local_correlation_matrix"], cmap="YlGnBu", ax=ax, cbar=False)
ax = sns.heatmap(adj_matrix_A, cmap="YlGnBu", ax=ax, cbar=False)
# colorbar
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()
# optimal_scale = metadata["optimal_scale"]
# ax.scatter(optimal_scale[0], optimal_scale[1], marker='X', s=200, color='red') 

ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Brain Regions(Nodes)', fontsize=15)
ax.set_ylabel('Brain Regions(Nodes)', fontsize=15) 
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
cax.xaxis.set_tick_params(labelsize=15)
cax.yaxis.set_tick_params(labelsize=15)

fig.suptitle('sub-NDARAE199TDD_acq-64dir_dwi_JHU_res-1x1x1_measure-spatial-ds.edgelist', fontsize=20)

plt.show()

# Define two rows for subplots
fig, (ax, cax) = plt.subplots(ncols=2, figsize=(9.45, 7.5),  gridspec_kw={"width_ratios":[1, 0.05]})
# Draw heatmap
# ax = sns.heatmap(metadata["local_correlation_matrix"], cmap="YlGnBu", ax=ax, cbar=False)
ax = sns.heatmap(adj_matrix_B, cmap="YlGnBu", ax=ax, cbar=False)
# colorbar
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()
# optimal_scale = metadata["optimal_scale"]
# ax.scatter(optimal_scale[0], optimal_scale[1], marker='X', s=200, color='red') 

ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Brain Regions(Nodes)', fontsize=15)
ax.set_ylabel('Brain Regions(Nodes)', fontsize=15) 
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
cax.xaxis.set_tick_params(labelsize=15)
cax.yaxis.set_tick_params(labelsize=15)

fig.suptitle('sub-NDARAJ366ZFA_acq-64dir_dwi_JHU_res-1x1x1_measure-spatial-ds.edgelist', fontsize=20)

plt.show()

# Define two rows for subplots
fig, (ax, cax) = plt.subplots(ncols=2, figsize=(9.45, 7.5),  gridspec_kw={"width_ratios":[1, 0.05]})
# Draw heatmap
ax = sns.heatmap(metadata["local_correlation_matrix"], cmap="YlGnBu", ax=ax, cbar=False)
# colorbar
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()
optimal_scale = metadata["optimal_scale"]
ax.scatter(optimal_scale[0], optimal_scale[1], marker='X', s=200, color='red') 

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('#Neighbors for sub-NDARAE199TDD', fontsize=15)
ax.set_ylabel('#Neighbors for sub-NDARAJ366ZFA', fontsize=15) 
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
cax.xaxis.set_tick_params(labelsize=15)
cax.yaxis.set_tick_params(labelsize=15)

fig.suptitle('cMGC = ' + str(metadata["test_statistic"]) + ', pMGC = ' + str(p_value), fontsize=20)

plt.show()