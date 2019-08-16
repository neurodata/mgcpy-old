import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from mgcpy.benchmarks.hypothesis_tests.three_sample_test.power import \
    power_given_epsilon
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.manova import Manova


def fill_params_dict_list_epsilons(base_path, do_fast_mgc=False):
    mcorr = DCorr(which_test='unbiased')
    mgc = MGC()
    manova = Manova()
    independence_tests = [manova, mcorr, mgc]
    three_sample_simulation_types = [1, 2, 3]

    params_dict_list = []
    for sim_type in three_sample_simulation_types:
        for test in independence_tests:
            params_dict = {'independence_test': test, 'simulation_type': sim_type, 'base_path': base_path, 'additional_params': {}}
            params_dict_list.append(params_dict)
        if do_fast_mgc:
            fast_mgc = MGC()
            additional_params = {"is_fast": True}
            params_dict = {'independence_test': fast_mgc, 'simulation_type': sim_type, 'base_path': base_path, 'additional_params': additional_params}
            params_dict_list.append(params_dict)
    return params_dict_list


def power_vs_epsilon_parallel(params_dict):
    '''
    Generate power of a 3 sample test given a simulation for a range of epsilon values
    '''
    epsilons = list(np.arange(0, 1, 0.05))
    estimated_power = np.zeros(len(epsilons))
    test = params_dict['independence_test']
    sim = params_dict['simulation_type']
    base_path = params_dict["base_path"]
    additional_params = params_dict['additional_params']
    if additional_params:
        test_name = 'fast_mgc'
    else:
        test_name = test.get_name()

    print(sim, test_name)

    for i in range(len(epsilons)):
        estimated_power[i] = power_given_epsilon(test, sim, epsilons[i], additional_params=additional_params)

    np.savetxt(os.path.join(base_path, 'python_power_curves_epsilon/{}_{}_epsilon.csv'.format(sim, test_name)), estimated_power, delimiter=',')
    print('{} {} finished'.format(sim, test_name))
    return (params_dict, estimated_power)


def plot_all_curves(base_path):
    simulation_names = ['Case 1', 'Case 2', 'Case 3']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    simulation_type = 0
    for i, col in enumerate(ax):
        sim_name = simulation_names[simulation_type]
        simulation_type += 1
        tests = ['mgc', 'unbiased', 'fast_mgc', 'manova']
        dir_name = os.path.join(base_path, 'python_power_curves_epsilon/')

        for test in tests:
            power = np.genfromtxt(dir_name + '{}_{}_epsilon.csv'.format(simulation_type, test), delimiter=',')
            x_axis = list(np.arange(0, 1, 0.05))
            col.plot(x_axis, power, label=test)
            # col.set_xlabel("epsilon")
            col.set_ylim(0, 1.2)
            col.set_yticks([0, 1])
            col.set_title(sim_name)

    plt.legend(bbox_to_anchor=(1.42, 0.5), loc="center right")
    plt.subplots_adjust(hspace=.75)

    fig.suptitle('Three Sample Test Power Curve for 3 Simulated 2-D Gaussians')
    plt.savefig(os.path.join(base_path, 'power_curves_epsilon'))


if __name__ == '__main__':
    base_path = "/root/code/mgcpy/benchmarks/hypothesis_tests/three_sample_test"

    start_time = time.time()
    params_dict = fill_params_dict_list_epsilons(base_path, do_fast_mgc=True)
    print('Finished filling params dict.')

    with mp.Pool(mp.cpu_count() - 1) as p:
        outputs = p.map(power_vs_epsilon_parallel, params_dict)

    print('All simulations and tests finished.')
    print('Took {} minutes'.format((time.time() - start_time) / 60))

    plot_all_curves(base_path)
