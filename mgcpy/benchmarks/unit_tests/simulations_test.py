import numpy as np
import pytest
from mgcpy.benchmarks import simulations as sims


def test_simulations():
    num_samps = 1000
    num_dim1 = 1
    num_dim2 = 300
    independent = True

    # Linear Simulation
    returns_low_dim = sims.linear_sim(num_samps, num_dim1)
    returns_high_dim = sims.linear_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Exponential Simulation
    returns_low_dim = sims.exp_sim(num_samps, num_dim1)
    returns_high_dim = sims.exp_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Cubic Simulation
    returns_low_dim = sims.cub_sim(num_samps, num_dim1)
    returns_high_dim = sims.cub_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Joint-Normal Simulation
    returns_low_dim = sims.joint_sim(num_samps, num_dim1)
    returns_high_dim = sims.joint_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Step Simulation
    returns_low_dim = sims.step_sim(num_samps, num_dim1)
    returns_high_dim = sims.step_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Quadratic Simulation
    returns_low_dim = sims.quad_sim(num_samps, num_dim1)
    returns_high_dim = sims.quad_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # W Simulation
    returns_low_dim = sims.w_sim(num_samps, num_dim1)
    returns_high_dim = sims.w_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Spiral Simulation
    returns_low_dim = sims.spiral_sim(num_samps, num_dim1)
    returns_high_dim = sims.spiral_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Uncorrelated Bernoulli Simulation
    returns = sims.ubern_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Logarithmic Simulation
    returns_low_dim = sims.log_sim(num_samps, num_dim1)
    returns_high_dim = sims.log_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Nth Root Simulation
    returns_low_dim = sims.root_sim(num_samps, num_dim1)
    returns_high_dim = sims.root_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Sinusoidal Simulation (4*pi)
    returns_low_dim = sims.sin_sim(num_samps, num_dim1)
    returns_high_dim = sims.sin_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Sinusoidal Simulation (16*pi)
    returns_low_dim = sims.sin_sim(num_samps, num_dim1, period=16*np.pi)
    returns_high_dim = sims.sin_sim(
        num_samps, num_dim2, period=16*np.pi, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Square Simulation
    returns = sims.square_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Two Parabolas Simulation
    returns_low_dim = sims.two_parab_sim(num_samps, num_dim1)
    returns_high_dim = sims.two_parab_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    # Circle Simulation
    returns = sims.circle_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Ellipse Simulation
    returns = sims.circle_sim(num_samps, num_dim2, radius=5)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Diamond Simulation
    returns = sims.square_sim(
        num_samps, num_dim2, period=-np.pi/4, indep=independent)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Multiplicative Noise Simulation
    returns = sims.multi_noise_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    # Multimodal Independence Simulation
    returns = sims.multi_indep_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))
