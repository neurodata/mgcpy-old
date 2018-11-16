import sys
module_path = "C:\\Users\\sunda\\Desktop\\AAA FA18 JHU\\NDD1\\gitscr"
if module_path not in sys.path:
    sys.path.append(module_path)
#import mgcpy.independence_tests
from mgcpy.independence_tests.mdmr.mdmr import MDMR
from mgcpy.independence_tests.mdmr.mdmrfunctions import compute_distance_matrix
from mgcpy.benchmarks.power import power
from mgcpy.benchmarks.simulations import *
import numpy as np
import pickle




def power_vs_dimension(independence_test, simulation_type, dim_range, sim_name):
    '''
    Generate power of an independence test given a simulation for a range of dimensions
    
    :param independence_test: an independence_test object
    :param simulation_type: a simulation function from mgcpy.benchmarks.simulations
    :param dim_range: the upper end of the range of dimension
    :return: power for each dimension
    '''
    estimated_power = np.zeros(dim_range)
    for d in range(1, dim_range+1):
        estimated_power[d-1] = power(independence_test, simulation_type, num_samples=100, num_dimensions=d, repeats=100, 
                                     simulation_type=sim_name)
    return estimated_power




def compute_all_power_vs_dimension(simulation_type, dim_range, sim_name):
    '''
    Compute power for each independence test for each dimension in a specified range 
    '''

    '''
    Initialize all independence test objects
    Data matrices initialized to np.nan, because in power computation each repeats involves generating new samples
    '''
    mdmr = MDMR(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix = compute_distance_matrix)
#    mcorr = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan,
#                  compute_distance_matrix=compute_distance_matrix, corr_type='mcorr')
#    dcorr = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan,
#                  compute_distance_matrix=compute_distance_matrix, corr_type='dcorr')
#    mantel = DCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan,
#                  compute_distance_matrix=compute_distance_matrix, corr_type='mantel')
#    mgc = MGC(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix)
#    rv_corr = RVCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix)
#    hhg = HHG(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix)
#    cca = RVCorr(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix=compute_distance_matrix, which_test='cca')
    
    
    independence_tests = {'MDMR': mdmr} 
#    {'MGC': mgc, 'MCorr': mcorr, 'DCorr': dcorr, 'Mantel': mantel,
#                          'RV Corr': rv_corr, 'CCA': cca} #, 'HHG': hhg}
    power_results = {}
    
    
    # compute power for each test for each dimension
    for name, test in independence_tests.items():
        power = power_vs_dimension(test, simulation_type, dim_range, sim_name)
        power_results[name] = power
        print('{} finished'.format(name))
    
    #independence_tests = [mcorr, dcorr]
    #power_results = Parallel(n_jobs=2)(iter[delayed(power_vs_dimension)(test, simulation_type, dim_range) for test in independence_tests])
    
    
    return power_results



def find_dim_range(sim_name):
    dim_range = 0
    if sim_name in ['joint_normal', 'sine_4pi', 'sine_16pi', 'multi_noise']:
        dim_range = 10
    elif sim_name in ['step', 'spiral', 'circle', 'ellipse', 'quadratic', 'w_shape', 'two_parabolas', 'fourth_root']:
        dim_range = 20
    elif sim_name in ['multi_indept', 'bernoulli', 'log']:
        dim_range = 100
    else:
        dim_range = 40
    return dim_range


simulations = {'joint_normal': joint_sim, 'sine_4pi': sin_sim, 'sine_16pi': sin_sim, 'multi_noise': multi_noise_sim,
               'step': step_sim, 'spiral': spiral_sim, 'circle': circle_sim, 'ellipse': circle_sim, 'diamond': square_sim,
               'log': log_sim, 'quadratic': quad_sim, 'w_shape': w_sim, 'two_parabolas': two_parab_sim, 'fourth_root': root_sim,
               'multi_indept': multi_indep_sim, 'bernoulli': ubern_sim}


for sim_name, sim_func in simulations.items():
    power_results = compute_all_power_vs_dimension(sim_func, find_dim_range(sim_name), sim_name)
    pickle.dump(power_results, open('C:/Users/sunda/Desktop/AAA FA18 JHU/NDD1/power_curve_{}.pkl'.format(sim_name), 'wb'))
    print('{} finished'.format(sim_name))



#csv1 = np.genfromtxt('X_mdmr.csv', delimiter=",")
#csv2 = np.genfromtxt('Y_mdmr.csv', delimiter=",")

#mdmr = MDMR(data_matrix_X=np.nan, data_matrix_Y=np.nan, compute_distance_matrix= compute_distance_matrix)
#mdmr_power = power(mdmr, w_sim, num_samples=100, num_dimensions=3, repeats = 100)
#print(mdmr_power)
#a = mdmr.test_statistic()[0]