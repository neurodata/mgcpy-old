
# coding: utf-8

import math
import multiprocessing as mp
import time

import numpy as np
import scipy.io
from numpy import genfromtxt

from mgcpy.benchmarks.power import power, power_given_data
from mgcpy.benchmarks.simulations import *
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.hhg import HHG
from mgcpy.independence_tests.kendall_spearman import KendallSpearman
from mgcpy.independence_tests.mdmr import MDMR
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr

# In[2]:



# In[5]:


def find_dim(sim_name):
    dim = 0
    if sim_name in ['joint_normal', 'sine_4pi', 'sine_16pi', 'multi_noise']:
        dim = 10
    elif sim_name in ['step', 'spiral', 'circle', 'ellipse', 'quadratic', 'w_shape', 'two_parabolas', 'fourth_root']:
        dim = 20
    elif sim_name in ['multi_indept', 'bernoulli', 'log']:
        dim = 100
    elif sim_name in ['linear', 'exponential', 'cubic']:
        dim = 1000
    else:
        dim = 40
    return dim


# In[6]:

simulations = {'joint_normal': (joint_sim, 4), 'sine_4pi': (sin_sim, 12), 'sine_16pi': (sin_sim, 13), 'multi_noise': (multi_noise_sim, 19),
               'step': (step_sim, 5), 'spiral': (spiral_sim, 8), 'circle': (circle_sim, 16), 'ellipse': (circle_sim, 17), 'diamond': (square_sim, 18),
               'log': (log_sim, 10), 'quadratic': (quad_sim, 6), 'w_shape': (w_sim, 7), 'two_parabolas': (two_parab_sim, 15), 'fourth_root': (root_sim, 11),
               'multi_indept': (multi_indep_sim, 20), 'bernoulli': (ubern_sim, 9), 'square': (square_sim, 14),
               'linear': (linear_sim, 1), 'exponential': (exp_sim, 2), 'cubic': (cub_sim, 3)}

# simulations = {'bernoulli': (ubern_sim, 9)}  # {'step': (step_sim, 5)}
# ## Parallel code

# In[10]:


def power_vs_dimension_parallel(params_dict):
    '''
    Generate power of an independence test given a simulation for a range of dimensions

    :param independence_test: an independence_test object
    :param simulation_type: a simulation function from mgcpy.benchmarks.simulations
    :param dim_range: the upper end of the range of dimension
    :return: power for each dimension
    '''
    test = params_dict['independence_test']
    sim = params_dict['simulation_type']
    additional_params = params_dict['additional_params']
    if additional_params:
        test_name = 'fast_mgc'
    else:
        test_name = test.get_name()

    print(sim, test_name)
    dim = params_dict['dim']
    if dim < 20:
        lim = 10
    else:
        lim = 20

    dim_range = np.arange(math.ceil(dim/lim), dim+1, math.ceil(dim/lim))
    if math.ceil(dim/lim) != 1:
        dim_range = np.insert(dim_range, 0, 1)
        lim = dim_range.shape[0]

    estimated_power = np.zeros(lim)
    for i in range(lim):
        estimated_power[i] = power_given_data(test, sim, num_samples=100, num_dimensions=dim_range[i], additional_params=additional_params)

    np.savetxt('../code/mgcpy/benchmarks/python_power_curves_dimensions/{}_{}_dimensions.csv'.format(sim, test_name), estimated_power, delimiter=',')
    print('{} {} finished'.format(sim, test_name))
    return (params_dict, estimated_power)


# In[11]:


def fill_params_dict_list_dimensions(do_fast_mgc=False):
    mcorr = DCorr(which_test='unbiased')
    dcorr = DCorr(which_test='biased')
    mantel = DCorr(which_test='mantel')
    mgc = MGC()
    rv_corr = RVCorr(which_test='rv')
    hhg = HHG()
    cca = RVCorr(which_test='cca')
    mdmr = MDMR()
    independence_tests = []  # [mgc, mcorr, dcorr, mantel, rv_corr, cca]

    params_dict_list = []
    for sim_name, sim_func in simulations.items():
        for test in independence_tests:
            params_dict = {'independence_test': test, 'simulation_type': sim_func[1], 'dim': find_dim(sim_name), 'additional_params': {}}
            params_dict_list.append(params_dict)
        if do_fast_mgc:
            fast_mgc = MGC()
            additional_params = {"is_fast": True}
            params_dict = {'independence_test': fast_mgc, 'simulation_type': sim_func[1], 'dim': find_dim(sim_name), 'additional_params': additional_params}
            params_dict_list.append(params_dict)

    return params_dict_list


# In[8]:


def power_vs_sample_size_parallel(params_dict):
    '''
    Generate power of an independence test given a simulation for a range of dimensions

    :param independence_test: an independence_test object
    :param simulation_type: a simulation function from mgcpy.benchmarks.simulations
    :param dim_range: the upper end of the range of dimension
    :return: power for each dimension
    '''
    sample_sizes = [i for i in range(5, 101, 5)]
    estimated_power = np.zeros(len(sample_sizes))
    test = params_dict['independence_test']
    sim = params_dict['simulation_type']
    additional_params = params_dict['additional_params']
    if additional_params:
        test_name = 'fast_mgc'
    else:
        test_name = test.get_name()

    print(sim, test_name)

    for i in range(len(sample_sizes)):
        # fast mgc doesn't work for less than 20 samples
        if test_name == 'fast_mgc' and sample_sizes[i] < 20:
            estimated_power[i] = np.nan
        else:
            estimated_power[i] = power_given_data(test, sim, data_type='sample_size', num_samples=sample_sizes[i], num_dimensions=1, additional_params=additional_params)

    np.savetxt('../code/mgcpy/benchmarks/python_power_curves_sample_size/{}_{}_sample_size.csv'.format(sim, test_name), estimated_power, delimiter=',')
    print('{} {} finished'.format(sim, test_name))
    return (params_dict, estimated_power)


# In[9]:


def fill_params_dict_list_sample_sizes(do_fast_mgc=False):
    mcorr = DCorr(which_test='unbiased')
    dcorr = DCorr(which_test='biased')
    mantel = DCorr(which_test='mantel')
    mgc = MGC()
    hhg = HHG()
    pearson = RVCorr(which_test='pearson')
    mdmr = MDMR()
    independence_tests = []

    params_dict_list = []
    for sim_name, sim_func in simulations.items():
        for test in independence_tests:
            params_dict = {'independence_test': test, 'simulation_type': sim_func[1], 'additional_params': {}}
            params_dict_list.append(params_dict)
        if do_fast_mgc:
            fast_mgc = MGC()
            additional_params = {"is_fast": True}
            params_dict = {'independence_test': fast_mgc, 'simulation_type': sim_func[1], 'additional_params': additional_params}
            params_dict_list.append(params_dict)
    return params_dict_list


# In[13]:

start_time = time.time()
params_dict = fill_params_dict_list_sample_sizes(do_fast_mgc=True)
# params_dict = fill_params_dict_list_dimensions(do_fast_mgc=True)
print('finished filling params dict')
# In[14]:


# pool = mp.Pool(mp.cpu_count()-1)
# results = pool.map(power_vs_dimension_parallel, params_dict)
with mp.Pool(mp.cpu_count() - 1) as p:
    outputs = p.map(power_vs_sample_size_parallel, params_dict)
    # outputs = p.map(power_vs_dimension_parallel, params_dict)

# power_vs_dimension_parallel(params_dict[0])
print('all finished')
print('took {} minutes'.format((time.time() - start_time) / 60))
