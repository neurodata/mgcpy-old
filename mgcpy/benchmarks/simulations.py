import numpy as np


def gen_coeffs(num_dim):
    """
    Helper function for generating a linear simulation.

    :param num_dim: number of dimensions for the simulation

    :return: a vector of coefficients
    """
    coeff_vec = np.array([1 / (x+1) for x in range(num_dim)])

    return coeff_vec.reshape(-1, 1)


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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = (np.matmul(a=x, b=coeffs) + kappa*noise*gauss_noise)
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = (np.exp(np.matmul(a=x, b=coeffs)) + kappa*noise*gauss_noise)
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    rho = 1 / (2*num_dim)
    sig = np.diag(np.ones(shape=(2*num_dim)) * (2*num_dim))
    sig[num_dim: (2*num_dim), 0: num_dim] = rho
    sig[0: num_dim, num_dim: (2*num_dim)] = rho

    samp = (np.random.multivariate_normal(cov=sig, mean=np.zeros(2*num_dim),
                                          size=num_samp))
    y = samp[:, num_dim: (2*num_dim)] + kappa*noise*gauss_noise
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    x_coeff = np.matmul(a=x, b=coeffs)
    x_coeff_temp = x_coeff.copy()
    x_coeff_temp[x_coeff < 0] = 0
    x_coeff_temp[x_coeff > 0] = 1
    y = (x_coeff_temp + kappa*noise*gauss_noise)
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = ((np.matmul(a=x, b=coeffs)**2) + kappa*noise*gauss_noise)
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
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0
    gauss_noise = np.random.normal(loc=0, scale=1, size=(x.shape[0], 1))

    y = (4 * ((np.matmul(a=x, b=coeffs)**2 - 0.5)**2
              + np.matmul(a=u, b=coeffs)/500)
         + kappa*noise*gauss_noise)
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)

    return x, y


def spiral_sim(num_samp, num_dim, noise=0.4, low=0, high=5):
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
    the_x = np.array(np.cos(np.pi * uniform_dist)).reshape(num_samp, 1)
    y = uniform_dist * np.sin(np.pi * uniform_dist)
    x = np.zeros(shape=(num_samp, num_dim))

    if num_dim > 1:
        for i in range(num_dim - 1):
            x[:, i] = np.squeeze((y * np.power(the_x, i)))
    x[:, num_dim-1] = np.squeeze(uniform_dist * the_x)

    gauss_noise = np.random.normal(loc=0, scale=1, size=(x.shape[0], 1))
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
    binom_dist = np.random.binomial(1, p=bern_prob, size=(num_samp, 1))
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

    gauss_noise2 = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    for i in range(num_samp):
        y[i] = (np.matmul((2*binom_dist[i]-1) * coeffs.T, x[i, :])
                + noise*gauss_noise2[i])

    return x, y


def log_sim(num_samp, num_dim, noise=1, indep=False, base=2):
    """
    Function for generating a logarithmic simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param base: the base of the log, defaults to 2

    :return: the data matrix and a response array
    """
    sig = np.diag(np.ones(shape=(num_dim)))
    x = (np.random.multivariate_normal(cov=sig, mean=np.zeros(num_dim),
                                       size=num_samp))
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = (base * np.divide(np.log(np.abs(x)), np.log(base)
                          + kappa*noise*gauss_noise))
    if indep:
        x = (np.random.multivariate_normal(cov=sig, mean=np.zeros(num_dim),
                                           size=num_samp))

    return x, y


def root_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1, n_root=4):
    """
    Function for generating an nth root simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param n_root: the root of the simulation, defaults to 4

    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = (np.power(np.abs(np.matmul(a=x, b=coeffs.reshape(num_dim, 1))), 1/n_root)
         + kappa*noise*gauss_noise/n_root)
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)

    return x, y
