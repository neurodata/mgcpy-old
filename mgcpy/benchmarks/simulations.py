#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gen_coeffs(num_dim):
    """
    Helper function for generating a linear simulation.

    :param num_dim: number of dimensions for the simulation

    :return: a vector of coefficients
    """
    coeff_vec = np.array([1 / (x+1) for x in range(num_dim)])
    
    return coeff_vec


def gen_x_unif(num_samp, num_dim, low=-1, high=1):
    """
    Helper function for generating n samples from d-dimensional vector

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
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    y = np.matmul(a=x, b=coeffs) + kappa*noise*gauss_noise
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


def exp_sim(num_samp, num_dim, noise=10, indep=False, low=0, high=3):
    """
    Function for generating an exponential simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 10
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to 0
    :param high: the upper limit of the data matrix, defaults to 3

    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    y = np.exp(np.matmul(a=x, b=coeffs)) + kappa*noise*gauss_noise
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


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
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    x_coeffs = np.matmul(a=x, b=coeffs)
    y = ((cub_coeff[2] * (x_coeffs-scale)**3)
        + (cub_coeff[1] * (x_coeffs-scale)**2)
        + (cub_coeff[0] * (x_coeffs-scale))
        + kappa * noise * gauss_noise)
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


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
    sig = np.diag(np.ones(shape=(2*num_dim)) * (2*num_dim))
    sig[num_dim : (2*num_dim), 0 : num_dim] = rho
    sig[0 : num_dim, num_dim : (2*num_dim)] = rho
    
    samp = (np.random.multivariate_normal(cov=sig, mean=np.zeros(2*num_dim), 
                                          size=num_samp))
    y = (samp[:, num_dim : (2*num_dim)] 
         + kappa*noise*gauss_noise.reshape(num_samp, 1))
    x = samp[:, 0:num_dim]
    
    return x, y


def step_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1):
    """
    Function for generating a joint-normal simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    
    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    x_coeff = np.matmul(a=x, b=coeffs)
    x_coeff_temp = x_coeff.copy()
    x_coeff_temp[x_coeff < 0] = 0
    x_coeff_temp[x_coeff > 0] = 1
    y = x_coeff_temp + kappa*noise*gauss_noise
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


def quad_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1):
    """
    Function for generating a quadratic simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    
    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    
    y = (np.matmul(a=x, b=coeffs)**2) + kappa*noise*gauss_noise
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


def w_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1):
    """
    Function for generating a w-shaped simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    
    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    u = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp,))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    gauss_noise = np.random.normal(loc=0, scale=1, size=(x.shape[0],))
    
    y = (4 * ((np.matmul(a=x, b=coeffs)**2 - 0.5)**2 
              + np.matmul(a=u, b=coeffs)/500) + kappa*noise*gauss_noise)
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    
    return x, y


def spiral_sim(num_samp, num_dim, noise=0.4, low=-0, high=5):
    """
    Function for generating a spiral simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 0.4
    :param low: the lower limit of the data matrix, defaults to 0
    :param high: the upper limit of the data matrix, defaults to 5
    
    :return: the data matrix and a response array
    """
    uniform_dist = gen_x_unif(num_samp, num_dim=1, low=low, high=high)
    x = np.array(np.cos(np.pi * uniform_dist)).reshape(num_samp, num_dim)
    y = uniform_dist * np.sin(np.pi * uniform_dist)
    
    if num_dim > 1:
        for i in range(num_dim - 1):
            x[:, i] = y * (x[:, i]) ** i
    x[:, num_dim-1] = uniform_dist.T * x[:, num_dim-1]
    
    gauss_noise = np.random.normal(loc=0, scale=1, size=(x.shape[0],))
    y = y + noise*num_dim*gauss_noise
    
    return x, y


def ubern_sim(num_samp, num_dim, noise=0.5, bern_prob=0.5):
    """
    Function for generating an uncorrelated bernoulli simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 0.5
    :param bern_prob: the bernoulli probability, defaults to 0.5
    
    :return: the data matrix and a response array
    """
    binom_dist = np.random.binomial(1, p=bern_prob, size=num_samp)
    sig = np.diag(np.ones(shape=num_dim) * num_dim)
    gauss_noise1 = (np.random.multivariate_normal(
                                                  cov=sig, 
                                                  mean=np.zeros(num_dim), 
                                                  size=num_samp
                                                  ))
    x = (np.array(np.random.binomial(1, size=num_samp * num_dim, 
                                     p=bern_prob)).reshape(num_samp, num_dim)
        + noise*gauss_noise1)
    
    coeffs = gen_coeffs(num_dim)
    y = np.empty(shape=(num_samp, 1))
    y[:] = np.nan
    
    gauss_noise2 = np.random.normal(loc=0, scale=1, size=(num_samp,))
    for i in range(num_samp):
        y[i] = (np.matmul((2*binom_dist[i]-1) * coeffs.T, x[i, :]) 
                + noise*gauss_noise2[i])
    
    return x, y

import matplotlib.pyplot as plt
returns = ubern_sim(100, 1, noise=0)
plt.plot(returns[0], returns[1], 'bo')


#def gen_sample_labels(num_class, class_equal=True):
#    """
#    Helper function for simulating sample labels
#
#    :param num_class: number of classes
#    :param class_equal: whether the number of samples/class should be equal, 
#                        with each class having a prior of 1/num_class, or 
#                        unequal, in which each class obtains a prior of 
#                        k/sum(num_class) for k in range(num_class), 
#                        defaults to true
#
#    :return: the class priors
#    """
#    sum = 0
#    if class_equal:
#        priors = np.array(1/num_class).reshape(num_class)
#    else:
#        for i in range(num_class):
#            sum += np.sum(i)
#            priors[i] = i / sum
#    
#    return priors
#
#
#def samp_rotate(dim):
#    """
#    Helper function for estimating a random rotation matrix.
#
#    :param d: number of dimensions to generate a rotation matrix for
#    
#    :return: the rotation matrix
#    """
#    rotate_matrix = np.linalg.qr(np.array(np.random.normal(size=(dim, dim))))
#    if (np.linalg.det(rotate_matrix) < -0.99):
#        rotate_matrix[:, 1] = -rotate_matrix[:, 1]
#        
#    return rotate_matrix
#
#
#def random_rotate(mus, covars, q=None):
#    """
#    Helper function for applying a random rotation to a gaussian parameter set.
#
#    :param mus: means per class
#    :param covars: covariances per class
#    :param q: rotation to use, defaults to none
#
#    :return: list of ``mus``, ``covars``, and ``q``
#    """
#    d, K = mus.shape
#    if (q == None):
#        q = samp_rotate(d)
#    try: 
#        q.shape = np.ones(shape=(d, d))
#    except:
#        print('You have specified a rotation matrix with dimensions (%d, %d), '
#              , q.shape[0], q.shape[1] + 'but should be (%d, %d).', d, d)
#    
#    for i in range(K):
#        mus[:, i] = np.matmul(q, mus[:, i])
#        covars[:, :, i] = np.matmul(q, np.matmul(covars[:, :, i], q.T))
#    
#    return [mus, covars, q]
#
#
#def sim_gmm(mus, sigmas, num_samp, priors):
#    """
#    Function to simulating from a guassian mixture
#
#    :param mus: the means for each class
#    :param sigmas: the variances for each class
#    :param num_samp: number of examples
#    :param priors: the priors for each class
#    
#    :return: a list containing the data matrix, response array, 
#             the sigma values, the priors, the simulation type, 
#             and the input parameters
#    """
#    k = mus.shape[1]
#    labs = np.random.choice(np.array(i for i in range(k)), size=num_samp, 
#                            p=priors)
#    ylabs = np.asarray(np.sort(np.unique(labs)))
#    res = 
#    
#    return [x, labs, priors]
#
#
#def disclin_sim(num_samp, num_dim, num_class, var_scale=1, mean_scale=1, 
#                rotate=False, class_equal=False, indep=False):
#    """
#    Function to simulate multi-class data with a linear class-mean trend.
#
#    :param num_samp: number of samples for the simulation
#    :param num_dim: number of dimensions for the simulation
#    :param num_class: number of classes in the dataset
#    :param var_scale: the scaling for the class-variance, defaults to 1
#    :param mean_scale: the scaling for the class-means, defaults to 1
#    :param rotate: the rotate scaling for the class, defaults to false
#    :param class_equal: whether the number of samples/class should be equal, 
#                        with each class having a prior of 1/num_class, or 
#                        unequal, in which each class obtains a prior of 
#                        k/sum(num_class) for k in range(num_class), 
#                        defaults to true
#    :param indep: whether to sample x and y independently, defaults to false
#    
#    :return: a list containing the data matrix, response array, 
#             the sigma values, the priors, the simulation type, 
#             and the input parameters
#    """
#    priors = gen_sample_labels(num_class, class_equal=class_equal)
#    sigma = var_scale * np.diag(num_dim)
#    sigma_multi = np.vstack(map(lambda i: sigma), range(num_class)[2]).T
#    
#    mu = np.array(1/np.sqrt(i) for i in range((i-1)**2 + 1)).reshspe(num_dim)
#    mus = np.vstack(map(lambda i: mu*i*mean_scale), range(num_class)[1]).T
#    
#    if (rotate):
#        mus, sigma = random_rotate(mus, sigma)
#    
#    x, y = sim_gmm(mus, sigma, num_samp, priors=priors)
#    
#    return [x, y]