from mgcpy.benchmarks import simulations as sims


def test_simulations():
    num_samps = 100
    num_dim1 = 1
    num_dim2 = 300
    independent = True

    assert sims.linear_sim(num_samps, num_dim1)
    assert sims.linear_sim(num_samps, num_dim2, indep=independent)

    assert sims.exp_sim(num_samps, num_dim1)
    assert sims.exp_sim(num_samps, num_dim2, indep=independent)

    assert sims.cub_sim(num_samps, num_dim1)
    assert sims.cub_sim(num_samps, num_dim2, indep=independent)

    assert sims.joint_sim(num_samps, num_dim1)
    assert sims.joint_sim(num_samps, num_dim2)

    assert sims.step_sim(num_samps, num_dim1)
    assert sims.step_sim(num_samps, num_dim2, indep=independent)

    assert sims.quad_sim(num_samps, num_dim1)
    assert sims.quad_sim(num_samps, num_dim2, indep=independent)

    assert sims.w_sim(num_samps, num_dim1)
    assert sims.w_sim(num_samps, num_dim2, indep=independent)

    assert sims.spiral_sim(num_samps, num_dim2)

    assert sims.ubern_sim(num_samps, num_dim1)
