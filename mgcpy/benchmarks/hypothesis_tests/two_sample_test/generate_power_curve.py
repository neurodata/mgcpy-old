import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from mgcpy.benchmarks.hypothesis_tests.two_sample_test.power import \
    power_given_data
from mgcpy.benchmarks.simulations import *
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.hhg import HHG
from mgcpy.independence_tests.mdmr import MDMR
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr

simulations = {'linear': (linear_sim, 1), 'exponential': (exp_sim, 2), 'cubic': (cub_sim, 3), 'joint_normal': (joint_sim, 4), 'step': (step_sim, 5),
               'quadratic': (quad_sim, 6), 'w_shape': (w_sim, 7), 'spiral': (spiral_sim, 8), 'bernoulli': (ubern_sim, 9), 'log': (log_sim, 10),
               'fourth_root': (root_sim, 11), 'sine_4pi': (sin_sim, 12), 'sine_16pi': (sin_sim, 13), 'square': (square_sim, 14), 'two_parabolas': (two_parab_sim, 15),
               'circle': (circle_sim, 16), 'ellipse': (circle_sim, 17), 'diamond': (square_sim, 18), 'multi_noise': (multi_noise_sim, 19), 'multi_indept': (multi_indep_sim, 20)}


def fill_params_dict_list_sample_sizes(base_path, do_fast_mgc=False):
    mcorr = DCorr(which_test='unbiased')
    dcorr = DCorr(which_test='biased')
    mantel = DCorr(which_test='mantel')
    mgc = MGC()
    hhg = HHG()
    pearson = RVCorr(which_test='pearson')
    mdmr = MDMR()
    independence_tests = [mcorr, dcorr, mantel, mgc, hhg, pearson, mdmr]

    params_dict_list = []
    for sim_name, sim_func in simulations.items():
        for test in independence_tests:
            params_dict = {'independence_test': test, 'simulation_type': sim_func[1], 'base_path': base_path, 'additional_params': {}}
            params_dict_list.append(params_dict)
        if do_fast_mgc:
            fast_mgc = MGC()
            additional_params = {"is_fast": True}
            params_dict = {'independence_test': fast_mgc, 'simulation_type': sim_func[1], 'base_path': base_path, 'additional_params': additional_params}
            params_dict_list.append(params_dict)
    return params_dict_list


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
    base_path = params_dict["base_path"]
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
            estimated_power[i] = power_given_data(base_path, test, sim, num_samples=sample_sizes[i], additional_params=additional_params)

    np.savetxt(os.path.join(base_path, 'python_power_curves_sample_size/{}_{}_sample_size.csv'.format(sim, test_name)), estimated_power, delimiter=',')
    print('{} {} finished'.format(sim, test_name))
    return (params_dict, estimated_power)


# for any additional test, add the name of the test (as defined in the `get_name` function in the class)
# in the list `tests` in the following function
def plot_all_curves(base_path):
    simulation_names = ['linear', 'exponential', 'cubic', 'joint_normal', 'step',
                        'quadratic', 'w_shape', 'spiral', 'bernoulli', 'log',
                        'fourth_root', 'sine_4pi', 'sine_16pi', 'square', 'two_parabolas',
                        'circle', 'ellipse', 'diamond', 'multi_noise', 'multi_indept']

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 12))
    simulation_type = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            sim_name = simulation_names[simulation_type]
            simulation_type += 1
            tests = ['mgc', 'unbiased', 'biased', 'mantel', 'pearson', 'mdmr', 'fast_mgc']
            dir_name = os.path.join(base_path, 'python_power_curves_sample_size/')

            for test in tests:
                power = np.genfromtxt(dir_name + '{}_{}_sample_size.csv'.format(simulation_type, test), delimiter=',')
                x_axis = [i for i in range(5, 101, 5)]
                # fast mgc is invalid for sample size less than 20
                if test == 'fast_mgc':
                    power[0:3] = np.nan
                col.plot(x_axis, power, label=test)
                # col.set_xlabel("num_samples")
                col.set_ylim(0, 1.2)
                col.set_yticks([0, 1])
                col.set_title(sim_name)

    plt.legend(bbox_to_anchor=(1.72, 0.5), loc="center right")
    plt.subplots_adjust(hspace=.75)

    fig.suptitle('Two Sample Test Power Curve for 20 Simulated 1-Dimensional Settings')
    plt.savefig(os.path.join(base_path, 'power_curves_sample_size'))


if __name__ == '__main__':
    base_path = "/root/code/mgcpy/benchmarks/hypothesis_tests/two_sample_test"

    start_time = time.time()
    params_dict = fill_params_dict_list_sample_sizes(base_path, do_fast_mgc=True)
    print('Finished filling params dict.')

    with mp.Pool(mp.cpu_count() - 1) as p:
        outputs = p.map(power_vs_sample_size_parallel, params_dict)

    print('All simulations and tests finished.')
    print('Took {} minutes'.format((time.time() - start_time) / 60))

    plot_all_curves(base_path)
