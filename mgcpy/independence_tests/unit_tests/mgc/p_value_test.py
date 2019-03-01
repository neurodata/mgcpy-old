import numpy as np
import pytest

from mgcpy.independence_tests.mgc import MGC


def test_mgc_test_linear():
    # linear (mgc.sims.linear(50, 1, 0.1))
    X = np.array([0.45233912,  0.41776082,  0.08992314, -0.68255391, -0.65492209,  0.24839759, -0.87725133,  0.32595345, -0.08646498, -0.16638085,  0.26394850,  0.72925148,  0.26028888, -0.59854218, -0.80068479, -0.69199885,  0.14915159,  0.37115868,  0.96039213,  0.27498675, -0.01372958, -0.89370963,  0.78395670, -0.42157105, -0.13893970,
                  0.50943310, -0.12623322, -0.20255325,  0.18437355, -0.02945578,  0.78082317,  0.39372362, -0.37730187, -0.17078540,  0.70129955,  0.83651364,  0.73375401, -0.34883304,  0.15323405,  0.51493599, -0.24317493,  0.83948953,  0.77216592,  0.90045095, -0.53736592, -0.88430486,  0.31447365,  0.66595322, -0.15917153, -0.38190466]).reshape(-1, 1)
    Y = np.array([0.665986696,  0.402397835,  0.134445492, -0.796653997, -0.636592886,  0.277283128, -0.636847542,  0.249515282, -0.149871134, -0.147567403,  0.369251601,  0.687118553,  0.524448340, -0.585999355, -0.858549573, -0.756081985,  0.129307735,  0.180976113,  0.874637167,  0.458794276, -0.003339139, -0.967879037,  0.758180626, -0.392856219, -0.114772505,
                  0.425345845, -0.069794980, -0.330857932,  0.229331072,  0.058739766,  0.777801029,  0.580715974, -0.231521102, -0.233366160,  0.669360658,  0.999785556,  0.648315305, -0.321119155,  0.156810807,  0.451349979, -0.393285002,  0.720164611,  0.811149183,  0.936183880, -0.587798720, -0.721394055,  0.233671350,  0.625407903, -0.154576153, -0.451475001]).reshape(-1, 1)

    p_value = 0

    mgc = MGC()
    p_value_res, _ = mgc.p_value(X, Y)
    assert np.allclose(p_value, p_value_res, rtol=0.1)


def test_mgc_test_non_linear():
    # spiral data (mgc.sims.spiral(50, 1, 0.5))
    X = np.array([-0.915363905,  2.134736725,  1.591825890, -0.947720469, -0.629203447,  0.157367412, -3.009624669,  0.342083914,  0.126834696,  2.009228424,  0.137638139, -4.168139174,  1.854371040,  1.696600346, -2.454855196,  1.770009913, -0.080973938,  1.985722698,  0.671279564,  1.521294941, -0.905490998, -1.043388333,  0.006493876,  4.007326886,  1.755316427, -
                  0.905436337,  0.497332481,  0.819071238,  3.561837453,  3.713293152,  0.487967353,  1.233385955, -2.985033861,  0.146394829, -2.231330093, -0.138580101, -2.390685794, -2.798259311,  0.647199716, -0.626705094, -0.254107788,  2.017131291, -2.871050739, -0.369874190,  0.198565130,  2.021387946, -2.877629992, -1.855015175, -0.201316471,  3.886001079]).reshape(-1, 1)
    Y = np.array([0.12441532, -2.63498763,  2.18349959, -0.58779997, -1.58602656,  0.35894756, -0.73954299,  1.76585591, -0.35002851,  0.48618590,  0.95628300,  1.99038991,  1.92277498,  1.34861841,  1.42509605,  0.65982368, -1.56731299, -0.17000082,  1.81187432, -0.73726241,  0.44491111,  0.19177688,  2.28190181,  0.45509215, -0.16777206,
                  0.06918430, -1.49570722,  2.23337087, -1.01335025, -0.60394315, -0.56653502, -3.12571299, -1.56146565,  0.52487563,  2.35561329, -1.79300788, -2.40650123,  0.53680541,  2.04171052,  0.09821259, -0.42712911,  0.52453433, -1.44426759, -2.22697039,  1.26906442, -0.13549404,  0.36776719, -2.44674330,  1.34647206,  2.14525574]).reshape(-1, 1)

    p_value = 0.7

    mgc = MGC()
    p_value_res, _ = mgc.p_value(X, Y)
    assert np.allclose(p_value, p_value_res, rtol=0.1)


def load_results(file_name, results_dir="./mgcpy/independence_tests/unit_tests/mgc/data/mgc/"):
    mgc_results = np.genfromtxt(results_dir + file_name, delimiter=',')[1:]

    pMGC = mgc_results[:, 0][0]
    statMGC = mgc_results[:, 1][0]
    pLocalCorr = mgc_results[:, 2:52]
    localCorr = mgc_results[:, 52:102]
    optimalScale = mgc_results[:, 102:104][0]

    return (pMGC, statMGC, pLocalCorr, localCorr, optimalScale)


def test_mgc_test_all():
    data_dir = "./mgcpy/independence_tests/unit_tests/mgc/data/"
    simulations = ["linear_sim", "exp_sim", "cub_sim", "joint_sim", "step_sim",
                   "quad_sim", "w_sim", "spiral_sim", "ubern_sim", "log_sim", "root_sim",
                   "sin_sim", "sin_sim_16", "square_sim", "two_parab_sim", "circle_sim",
                   "ellipsis_sim", "square_sim_", "multi_noise_sim", "multi_indep_sim"]

    print("\nSimulations being used to test MGC: ")
    for simulation in simulations:
        print(simulation)

        X = np.genfromtxt(data_dir + "input/" + simulation + "_x.csv", delimiter=',').reshape(-1, 1)
        Y = np.genfromtxt(data_dir + "input/" + simulation + "_y.csv", delimiter=',').reshape(-1, 1)

        if simulation == "step_sim":
            mgc_results = np.genfromtxt(data_dir + "mgc/" + simulation + "_res.csv", delimiter=',')[1:]
            pMGC = mgc_results[:, 0][0]
            statMGC = mgc_results[:, 1][0]
            # pLocalCorr = mgc_results[:, 2:4]
            localCorr = mgc_results[:, 4:6]
            optimalScale = mgc_results[:, 6:8][0]
        else:
            pMGC, statMGC, _, localCorr, optimalScale = load_results(simulation + "_res.csv")

        mgc = MGC()
        p_value, metadata = mgc.p_value(X, Y)

        assert np.allclose(statMGC, metadata["test_statistic"])
        assert np.allclose(localCorr, metadata["local_correlation_matrix"])
        assert np.allclose(optimalScale, metadata["optimal_scale"])
        assert np.allclose(pMGC, p_value, atol=0.1)
