from mgcpy.independence_tests.mgc import MGC
from mgcpy.benchmarks.simulations import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import multiprocessing as mp
from stats import multiscale_graphcorr
import os
import scipy.io
from sklearn.ensemble import RandomForestRegressor
from mgcpy.hypothesis_tests.transforms import k_sample_transform
import math
import numpy as np
import sys
sys.path.append("~/mgcpy")
sys.path.append("~/scipy")
sys.path.append("~/scipy/scipy")
sys.path.append("~/scipy/scipy/stats")


def proximityMatrix(model, X, normalize=True):

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


def power_scipy(base_path, simulation_type, num_samples, repeats=1000, alpha=.05):
    # direct p values on permutation
    p_values = np.zeros(repeats)

    # absolute path to the benchmark directory
    file_name_prefix = os.path.join(base_path, 'sample_data_power_sample_sizes/type_{}_size_{}'.format(simulation_type, num_samples))

    all_matrix_X = scipy.io.loadmat(file_name_prefix + '_X.mat')['x_mtx'][..., np.newaxis]
    all_matrix_Y = scipy.io.loadmat(file_name_prefix + '_Y.mat')['y_mtx'][..., np.newaxis]

    # rotation transform matrix
    c, s = np.cos(math.radians(60)), np.sin(math.radians(60))
    rotation_matrix = np.array([[c, s], [-s, c]])

    for rep in range(repeats):
        matrix_X = all_matrix_X[rep, :, :]
        matrix_Y = all_matrix_Y[rep, :, :]

        # apply two sample transform
        data_matrix = np.concatenate([matrix_X, matrix_Y], axis=1)
        rotated_data_matrix = np.dot(rotation_matrix, data_matrix.T).T
        matrix_U, matrix_V = k_sample_transform(data_matrix, rotated_data_matrix)

        rf_matrix_V = matrix_V.reshape(-1)
        clf = RandomForestRegressor(n_estimators=500)
        clf.fit(matrix_U, rf_matrix_V)
        matrix_U = 1 - proximityMatrix(clf, matrix_U, normalize=True)
        matrix_U = np.power(matrix_U, 0.5)

        mgc = multiscale_graphcorr(matrix_U, matrix_V)
        p_values[rep] = mgc.pvalue

    empirical_power = np.where(p_values <= alpha)[0].shape[0] / repeats

    return empirical_power


# from mgcpy.benchmarks.hypothesis_tests.two_sample_test.power import power_scipy
# from mgcpy.independence_tests.dcorr import DCorr
# from mgcpy.independence_tests.hhg import HHG
# from mgcpy.independence_tests.mdmr import MDMR
# from mgcpy.independence_tests.rv_corr import RVCorr
sns.color_palette('Set1')
sns.set(color_codes=True, style='white', context='talk', font_scale=1.9)


simulations = {'linear': (linear_sim, 1),
               'exponential': (exp_sim, 2),
               'cubic': (cub_sim, 3),
               'joint_normal': (joint_sim, 4),
               'step': (step_sim, 5),
               'quadratic': (quad_sim, 6),
               'w_shape': (w_sim, 7),
               'spiral': (spiral_sim, 8),
               'bernoulli': (ubern_sim, 9),
               'log': (log_sim, 10),
               'fourth_root': (root_sim, 11),
               'sine_4pi': (sin_sim, 12),
               'sine_16pi': (sin_sim, 13),
               'square': (square_sim, 14),
               'two_parabolas': (two_parab_sim, 15),
               'circle': (circle_sim, 16),
               'ellipse': (circle_sim, 17),
               'diamond': (square_sim, 18),
               'multi_noise': (multi_noise_sim, 19),
               'multi_indept': (multi_indep_sim, 20)
               }

plot_titles = ['Linear',
               'Exponential',
               'Cubic',
               'Joint Normal',
               'Step',
               'Quadratic',
               'W-Shaped',
               'Spiral',
               'Uncorrelated\nBernoulli',
               'Logarithmic',
               'Fourth Root',
               'Sine (4$\pi$)',
               'Sine (16$\pi$)',
               'Square',
               'Two Parabolas',
               'Circle',
               'Ellipse',
               'Diamond',
               'Multiplicative\nNoise',
               'Multimodal\nIndependence'
               ]


def fill_params_dict_list_sample_sizes(base_path):
    independence_tests = ['mgc_rf']  # [mcorr, dcorr, mantel, mgc, hhg, pearson, mdmr]

    params_dict_list = []
    for sim_name, sim_func in simulations.items():
        for test in independence_tests:
            params_dict = {'independence_test': test, 'simulation_type': sim_func[1], 'base_path': base_path}
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
    sim = params_dict["simulation_type"]
    base_path = params_dict["base_path"]
    test_name = params_dict["independence_test"]

    print(sim, test_name)

    for i in tqdm(range(len(sample_sizes))):
        estimated_power[i] = power_scipy(base_path, sim, num_samples=sample_sizes[i])

    np.savetxt(os.path.join(base_path, 'python_power_curves_sample_size/{}_{}_sample_size.csv'.format(sim, test_name)), estimated_power, delimiter=',')
    print('{} {} finished'.format(sim, test_name))
    return (params_dict, estimated_power)


# for any additional test, add the name of the test (as defined in the `get_name` function in the class)
# in the list `tests` in the following function
def plot_all_curves(base_path):
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(28, 24), sharex=True, sharey=True)
    simulation_type = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            tests = ['mgc', 'unbiased', 'biased', 'mantel', 'pearson', 'mdmr', 'fast_mgc', 'cca', 'mgc_rf']
            test_names = ['MGC', 'Unbiased Dcorr', 'Biased Dcorr', 'Mantel', 'Pearson', 'MDMR', 'Fast MGC', 'MANOVA', 'MGC RF']
            dir_name = os.path.join(base_path, 'python_power_curves_sample_size/')

            for test_num, test in enumerate(tests):
                power = np.genfromtxt(dir_name + '{}_{}_sample_size.csv'.format(simulation_type+1, test), delimiter=',')
                mgc_power = np.genfromtxt(dir_name + '{}_mgc_sample_size.csv'.format(simulation_type+1), delimiter=',')

                x_axis = [i for i in range(5, 101, 5)]

                # fast mgc is invalid for sample size less than 20
                if test == 'fast_mgc':
                    power[0:3] = np.nan

                # power = power - mgc_power
                if test == 'mgc':
                    col.plot(x_axis, power - mgc_power, label=test_names[test_num], lw=4, color='red')
                elif test == 'fast_mgc':
                    col.plot(x_axis, power - mgc_power, label=test_names[test_num], lw=4, color='red', linestyle=':')
                elif test == 'unbiased':
                    col.plot(x_axis, power - mgc_power, label=test_names[test_num], lw=3, color='blue')
                else:
                    col.plot(x_axis, power - mgc_power, label=test_names[test_num], lw=3)

                col.set_title(plot_titles[simulation_type], fontsize=35)
                col.set_xticks([x_axis[0], x_axis[-1]])
                col.set_ylim(-1.1, 1.1)
                col.set_yticks([-1, 0, 1])
                #col.set_ylim(0, 1.1)
                #col.set_yticks([0, 1])

            simulation_type += 1

    leg = plt.legend(bbox_to_anchor=(0.5, 0.08), bbox_transform=plt.gcf().transFigure, ncol=5, loc='upper center')
    leg.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(hspace=.65)

    fig.text(0.5, 0.08, 'Sample Size', ha='center')
    fig.text(0.08, 0.5, 'Power Relative to MGC', va='center', rotation='vertical')

    # fig.suptitle('Two Sample Test Power Curve for 20 Simulated 1-Dimensional Settings')
    plt.savefig(os.path.join(base_path, 'two_sample_power_vs_sample_size'), bbox_inches='tight')
    plt.savefig(os.path.join(base_path, 'two_sample_power_vs_sample_size.eps'), bbox_inches='tight')


if __name__ == '__main__':
    base_path = "~/mgcpy/mgcpy/benchmarks/hypothesis_tests/two_sample_test"
    # base_path = "/Users/pikachu/OneDrive - Johns Hopkins University/Mac Desktop/NDD I/mgcpy/mgcpy/benchmarks/hypothesis_tests/two_sample_test"

    start_time = time.time()
    params_dict = fill_params_dict_list_sample_sizes(base_path)
    print('Finished filling params dict.')

    with mp.Pool(mp.cpu_count() - 1) as p:
        outputs = p.map(power_vs_sample_size_parallel, params_dict)

    print('All simulations and tests finished.')
    print('Took {} minutes'.format((time.time() - start_time) / 60))

    plot_all_curves(base_path)
