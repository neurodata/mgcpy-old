#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gen_coeffs(num_dim):
    """
    Function for generating a linear simulation.

    :param num_dim: number of dimensions for the simulation

    :return: a vector of coefficients
    """
    coeff_vec = np.array([1/(x+1) for x in range(num_dim)])
    
    return coeff_vec


def gen_x_unif(num_samp, num_dim, low=-1, high=1):
    """
    Function for generating n samples from d-dimensional vector

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1

    :return: uniformly distributed simulated data matrix
    """
    uniform_vec = np.array(np.random.uniform(low=low, high=high, 
                                             size=num_samp * num_dim))
    data_mat = uniform_vec.reshape(num_samp, num_dim)
    
    return data_mat


def linear_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1):
    """
    Function for generating a linear simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1

    :return: the data matrix and a response array
    """
    dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    y = np.matmul(a=dist, b=coeffs.T) + kappa * noise * gauss_noise
    
    return dist, y

A = linear_sim(100, 1)

import matplotlib.pyplot as plt
plt.plot(A[0], A[1], '.')
plt.axis('off')