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

    y = (np.dot(a=x, b=coeffs) + kappa*noise*gauss_noise)
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

    y = (np.exp(np.dot(a=x, b=coeffs)) + kappa*noise*gauss_noise)
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

    x_coeffs = np.dot(a=x, b=coeffs)
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
    if (num_dim > 1):
        kappa = 1
    else:
        kappa = 0
    rho = 1 / (2*num_dim)
    sig = np.diag(np.ones(shape=(2*num_dim)))
    sig[num_dim: (2*num_dim), 0: num_dim] = rho
    sig[0: num_dim, num_dim: (2*num_dim)] = rho

    samp = (np.random.multivariate_normal(cov=sig, mean=np.zeros(2*num_dim),
                                          size=num_samp))
    if num_dim == 1:
        y = samp[:, (num_dim):(2*num_dim)] + kappa*noise*gauss_noise
        x = samp[:, 0:num_dim]
    else:
        y = samp[:, (num_dim+1):(2*num_dim)] + kappa*noise*gauss_noise
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
    if (num_dim > 1):
        kappa = 1
    else:
        kappa = 0

    x_coeff = np.dot(a=x, b=coeffs)
    x_coeff = 1 * (x_coeff > 0)
    y = (x_coeff + kappa*noise*gauss_noise)
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

    y = ((np.dot(a=x, b=coeffs)**2) + kappa*noise*gauss_noise)
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

    y = (4 * ((np.dot(a=x, b=coeffs)**2 - 0.5)**2 + np.dot(a=u, b=coeffs)/500)
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
    if num_dim > 1:
        kappa = 1
    else:
        kappa = 0
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
        y[i] = (np.dot((2*binom_dist[i]-1) * coeffs.T, x[i, :])
                + kappa*noise*gauss_noise2[i])

    return x, y


def log_sim(num_samp, num_dim, noise=3, indep=False, base=2):
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


def root_sim(num_samp, num_dim, noise=0.25, indep=False, low=-1, high=1, n_root=4):
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

    y = (np.power(np.abs(np.dot(a=x, b=coeffs.reshape(num_dim, 1))), 1/n_root)
         + kappa*noise*gauss_noise)
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)

    return x, y


def sin_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1, period=4*np.pi):
    """
    Function for generating a sinusoid simulation.

    Note: For producing 4*pi and 16*pi simulations, change the ``period`` to the respective value.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 1
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param period: the period of the sine wave, defaults to 4*pi

    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    if num_dim > 1 or noise > 0:
        sig = np.diag(np.ones(shape=(num_dim)))
        v = (np.random.multivariate_normal(cov=sig, mean=np.zeros(num_dim),
                                           size=num_samp))
        x = x + 0.02*num_dim*v
    gauss_noise = np.random.normal(loc=0, scale=1, size=(num_samp, 1))
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = np.sin(x*period) + kappa*noise*gauss_noise
    if indep:
        x = gen_x_unif(num_samp, num_dim, low=low, high=high)
        if num_dim > 1:
            sig = np.diag(np.ones(shape=(num_dim)))
            v = (np.random.multivariate_normal(cov=sig, mean=np.zeros(num_dim),
                                               size=num_samp))
            x = x + 0.02*num_dim*v

    return x, y


def square_sim(num_samp, num_dim, noise=1, indep=False, low=-1, high=1, period=-np.pi/8):
    """
    Function for generating a square or diamond simulation.

    Note: For producing square or diamond simulations, change the ``period`` to -pi/8 or -pi/4.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 0.05
    :param indep: whether to sample x and y independently, defaults to false
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param period: the period of the sine and cosine square equation, defaults to 4*pi

    :return: the data matrix and a response array
    """
    u = gen_x_unif(num_samp, num_dim, low=low, high=high)
    v = gen_x_unif(num_samp, num_dim, low=low, high=high)
    sig = np.diag(np.ones(shape=(num_dim)))
    gauss_noise = (np.random.multivariate_normal(cov=sig,
                                                 mean=np.zeros(num_dim),
                                                 size=num_samp))
    x = u*np.cos(period) + v*np.sin(period) + 0.05*num_dim*gauss_noise

    y = -u*np.sin(period) + v*np.cos(period)
    if indep:
        u = gen_x_unif(num_samp, num_dim, low=low, high=high)
        v = gen_x_unif(num_samp, num_dim, low=low, high=high)
        sig = np.diag(np.ones(shape=(num_dim)))
        gauss_noise = (np.random.multivariate_normal(cov=sig,
                                                     mean=np.zeros(
                                                         num_dim),
                                                     size=num_samp))
        x = u*np.cos(period) + v*np.sin(period) + 0.05*num_dim*gauss_noise

    return x, y


def two_parab_sim(num_samp, num_dim, noise=2, low=-1, high=1, prob=0.5):
    """
    Function for generating a two parabolas simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 2
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param prob: the binomial probability, defaults to 0.5

    :return: the data matrix and a response array
    """
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    coeffs = gen_coeffs(num_dim)
    u = np.random.binomial(1, p=prob, size=(num_samp, 1))
    gauss_noise = gen_x_unif(num_samp, num_dim, low=0, high=1)
    if (num_dim == 1):
        kappa = 1
    else:
        kappa = 0

    y = (np.power(np.dot(x, coeffs.reshape(num_dim, 1)), 2) +
         noise*kappa*gauss_noise) * (u - 0.5)

    return x, y


def circle_sim(num_samp, num_dim, noise=0.4, low=-1, high=1, radius=1):
    """
    Function for generating a circle or ellipse simulation.

    Note: For producing circle or ellipse simulations, change the ``radius`` to 1 or 5.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param noise: noise level of the simulation, defaults to 0.4
    :param low: the lower limit of the data matrix, defaults to -1
    :param high: the upper limit of the data matrix, defaults to 1
    :param radius: the radius of the circle or ellipse, defaults to 1

    :return: the data matrix and a response array
    """
    if num_dim > 1:
        kappa = 1
    else:
        kappa = 0
    x = gen_x_unif(num_samp, num_dim, low=low, high=high)
    rx = radius * np.ones((num_samp, num_dim))
    z = gen_x_unif(num_samp, num_dim, low=low, high=high)
    sig = np.diag(np.ones(shape=(num_dim)))
    gauss_noise = (np.random.multivariate_normal(cov=sig,
                                                 mean=np.zeros(num_dim),
                                                 size=num_samp))

    ry = np.ones((num_samp, num_dim))
    x[:, 0] = np.cos(z[:, 0].reshape((num_samp)) * np.pi)
    for i in range(num_dim - 1):
        x[:, i+1] = (x[:, i].reshape((num_samp)) * np.cos(z[:, i+1].reshape((num_samp)) * np.pi))
        x[:, i] = (x[:, i].reshape((num_samp)) * np.sin(z[:, i+1].reshape((num_samp)) * np.pi))
    x = rx * x + noise*rx*gauss_noise

    y = ry * np.sin(z[:, 0].reshape((num_samp, 1)) * np.pi)

    return x, y


def multi_noise_sim(num_samp, num_dim):
    """
    Function for generating a multiplicative noise simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation

    :return: the data matrix and a response array
    """
    sig = np.diag(np.ones(shape=(num_dim)))
    u = np.random.multivariate_normal(
        cov=sig, mean=np.zeros(num_dim), size=num_samp)
    x = np.random.multivariate_normal(
        cov=sig, mean=np.zeros(num_dim), size=num_samp)

    y = u * x

    return x, y


def multi_indep_sim(num_samp, num_dim, prob=0.5, sep1=3, sep2=2):
    """
    Function for generating a multimodal independence simulation.

    :param num_samp: number of samples for the simulation
    :param num_dim: number of dimensions for the simulation
    :param prob: the binomial probability, defaults to 0.5
    :param sep1: determines the size and separation of clusters, defaults to 3
    :param sep2: determines the size and separation of clusters, defaults to 2

    :return: the data matrix and a response array
    """
    sig = np.diag(np.ones(shape=(num_dim)))
    u = np.random.multivariate_normal(
        cov=sig, mean=np.zeros(num_dim), size=num_samp)
    v = np.random.multivariate_normal(
        cov=sig, mean=np.zeros(num_dim), size=num_samp)
    u_2 = np.random.binomial(1, p=prob, size=(num_samp, 1))
    v_2 = np.random.binomial(1, p=prob, size=(num_samp, 1))

    x = u/sep1 + sep2*u_2 - 1
    y = v/sep1 + sep2*v_2 - 1

    return x, y
