import numpy as np
from mgcpy.benchmarks.simulations import (circle_sim, cub_sim, exp_sim,
                                          joint_sim, linear_sim, log_sim,
                                          multi_indep_sim, multi_noise_sim,
                                          quad_sim, root_sim, sin_sim,
                                          spiral_sim, square_sim, step_sim,
                                          two_parab_sim, ubern_sim, w_sim)


def gen_data(data_dir="./mgcpy/independence_tests/unit_tests/mgc/data/input/"):
    NUM_SAMPLES = 50
    NUM_DIMS = 1

    def sin_sim_16(x, y): return sin_sim(x, y, period=16*np.pi)

    def ellipsis_sim(x, y): return circle_sim(x, y, radius=5)

    def square_sim_(x, y): return square_sim(x, y, period=-np.pi/4, indep=True)

    simulations = [linear_sim, exp_sim, cub_sim, joint_sim, step_sim,
                   quad_sim, w_sim, spiral_sim, ubern_sim, log_sim,
                   root_sim, sin_sim, sin_sim_16, square_sim, two_parab_sim,
                   circle_sim, ellipsis_sim, square_sim_, multi_noise_sim, multi_indep_sim]

    for simulation in simulations:
        x, y = simulation(NUM_SAMPLES, NUM_DIMS)
        np.savetxt(data_dir + str(simulation.__name__) + "_x.csv", x, delimiter=",")
        np.savetxt(data_dir + str(simulation.__name__) + "_y.csv", y, delimiter=",")


def load_results(file_name, results_dir="./mgcpy/independence_tests/unit_tests/mgc/data/mgc/"):
    mgc_results = np.genfromtxt(results_dir + file_name, delimiter=',')[1:]

    pMGC = mgc_results[:, 0][0]
    statMGC = mgc_results[:, 1][0]
    pLocalCorr = mgc_results[:, 2:52]
    localCorr = mgc_results[:, 52:102]
    optimalScale = mgc_results[:, 102:104][0]

    return (pMGC, statMGC, pLocalCorr, localCorr, optimalScale)


if __name__ == '__main__':
    gen_data()
    # print(load_results("linear_sim_res.csv"))
