#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gen_coeffs(num_dim):
    """
    Function for generating a linear simulation.

    :param num_dim: number of dimensions for the simulation

    :return: a vector of coefficients
    """
    coeff_vec = np.array([1 / (x+1) for x in range(num_dim)])
    
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
    
    y = np.matmul(a=dist, b=coeffs.T) + kappa*noise*gauss_noise
    if indep:
        dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return dist, y


def exp_sim(num_samp, num_dim, noise=10, indep=False, low=0, high=3):
    """
    Function for generating a exponential simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 10
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to 0
    :param high: the upper limit of the data matrix, defaults to 3

    :return: the data matrix and a response array
    """
    dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    y = np.exp(np.matmul(a=dist, b=coeffs.T)) + kappa*noise*gauss_noise
    if indep:
        dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return dist, y


def cub_sim(num_samp, num_dim, noise=80, indep=False, low=-1, high=1,
            cub_coeff=np.array([-12, 48, 128]), scale=1/3):
    """
    Function for generating a cubic simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 80
    :param indep: whether to sample x and y independently, defaults to False
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param cub_coeff: coefficients of the cubic function where each value
                      corresponds to the respective order coefficientj,
                      defaults to [-12, 48, 128]
    :param scale: scaling center of the cubic, defaults to 1/3

    :return: the data matrix and a response array
    """
    dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    dist_coeffs = np.matmul(a=dist, b=coeffs.T)
    y = ((cub_coeff[2] * (dist_coeffs-scale) ** 3)
        + (cub_coeff[1] * (dist_coeffs-scale) ** 2)
        + (cub_coeff[0] * (dist_coeffs-scale))
        + kappa*noise*gauss_noise)
    if indep:
        dist = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return dist, y


def joint_sim(num_samp, num_dim, noise=0.5):
    """
    Function for generating a joint-normal simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 80

    :return: the data matrix and a response array
    """
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    rho = 1 / (2*num_dim)
    sig = np.diag(2*num_dim)
    sig[num_dim : (2*num_dim), 0 : num_dim] = rho
    sig[0 : num_dim, num_dim : (2*num_dim)] = rho
    
    samp = (np.random.multivariate_normal(cov=sig, mu=np.zeros(2*num_dim))
            * num_samp)
    y = samp[:, num_dim : (2*num_dim)] + kappa*noise*gauss_noise
    x = samp[:, 0:num_dim]
    
    return x, y