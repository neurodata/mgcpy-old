import pickle

import h5py
import numpy as np
import pytest
from mgcpy.benchmarks.power import power, power_given_data
from mgcpy.benchmarks.simulations import (circle_sim, joint_sim,
                                          multi_indep_sim, multi_noise_sim,
                                          sin_sim, square_sim, ubern_sim,
                                          w_sim)
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.rv_corr import RVCorr
from scipy.spatial.distance import pdist, squareform


def test_power():
    test = DCorr(which_test='unbiased')
    simulation_type = 4
    sim = joint_sim
    sample_sizes = [i for i in range(5, 101, 5)]

    matlab_file_name = './mgcpy/benchmarks/matlab_power_results/sample_size/CorrIndTestType{}N100Dim1.mat'.format(simulation_type)
    with h5py.File(matlab_file_name, 'r') as f:
        matlab_results = {}
        for k, v in f.items():
            matlab_results[k] = np.transpose(np.array(v))
    matlab_power = matlab_results['powerM'][0, :]

    estimated_power = np.zeros(len(sample_sizes))
    for i in range(len(sample_sizes)):
        estimated_power[i] = power(test, sim, num_samples=sample_sizes[i], num_dimensions=1)
    assert np.allclose(estimated_power, matlab_power, atol=0.2)


def test_power_given_data():
    test = DCorr(which_test='unbiased')
    simulation_type = 4
    sample_sizes = [i for i in range(5, 101, 5)]

    matlab_file_name = './mgcpy/benchmarks/matlab_power_results/sample_size/CorrIndTestType{}N100Dim1.mat'.format(simulation_type)
    with h5py.File(matlab_file_name, 'r') as f:
        matlab_results = {}
        for k, v in f.items():
            matlab_results[k] = np.transpose(np.array(v))
    matlab_power = matlab_results['powerM'][0, :]

    estimated_power = np.zeros(len(sample_sizes))
    for i in range(len(sample_sizes)):
        estimated_power[i] = power_given_data(test, simulation_type, data_type='sample_size', num_samples=sample_sizes[i], num_dimensions=1)
    assert np.allclose(estimated_power, matlab_power, atol=0.2)
