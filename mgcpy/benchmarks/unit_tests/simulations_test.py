import matplotlib.pyplot as plt
import numpy as np
from mgcpy.benchmarks import simulations as sims
from mpl_toolkits.mplot3d import Axes3D


def test_simulations():
    num_samps = 1000
    num_dim1 = 1
    num_dim2 = 300
    independent = True

    np.random.seed(0)
    fig1 = plt.figure(figsize=(50, 80))
    plt.axis('off')
    fig2 = plt.figure(figsize=(50, 80))

    # Linear Simulation
    returns_low_dim = sims.linear_sim(num_samps, num_dim1)
    returns_high_dim = sims.linear_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.linear_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 1)
    ax1.scatter(x1, y1)
    ax1.set_title('Linear', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.linear_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 1, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Linear', fontweight='bold')
    ax2.axis('off')

    # Exponential Simulation
    returns_low_dim = sims.exp_sim(num_samps, num_dim1)
    returns_high_dim = sims.exp_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.exp_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 2)
    ax1.scatter(x1, y1)
    ax1.set_title('Exponential', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.exp_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 2, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Exponential', fontweight='bold')
    ax2.axis('off')

    # Cubic Simulation
    returns_low_dim = sims.cub_sim(num_samps, num_dim1)
    returns_high_dim = sims.cub_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.cub_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 3)
    ax1.scatter(x1, y1)
    ax1.set_title('Cubic', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.cub_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 3, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Cubic', fontweight='bold')
    ax2.axis('off')

    # Joint-Normal Simulation
    returns_low_dim = sims.joint_sim(num_samps, num_dim1)
    returns_high_dim = sims.joint_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.joint_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 4)
    ax1.scatter(x1, y1)
    ax1.set_title('Joint Normal', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.joint_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 4, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Joint Normal', fontweight='bold')
    ax2.axis('off')

    # Step Simulation
    returns_low_dim = sims.step_sim(num_samps, num_dim1)
    returns_high_dim = sims.step_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.step_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 5)
    ax1.scatter(x1, y1)
    ax1.set_title('Step', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.step_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 5, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Step', fontweight='bold')
    ax2.axis('off')

    # Quadratic Simulation
    returns_low_dim = sims.quad_sim(num_samps, num_dim1)
    returns_high_dim = sims.quad_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.quad_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 6)
    ax1.scatter(x1, y1)
    ax1.set_title('Quadratic', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.quad_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 6, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Quadratic', fontweight='bold')
    ax2.axis('off')

    # W Simulation
    returns_low_dim = sims.w_sim(num_samps, num_dim1)
    returns_high_dim = sims.w_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.w_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 7)
    ax1.scatter(x1, y1)
    ax1.set_title('W-Shaped', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.w_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 7, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('W-Shaped', fontweight='bold')
    ax2.axis('off')

    # Spiral Simulation
    returns_low_dim = sims.spiral_sim(num_samps, num_dim1)
    returns_high_dim = sims.spiral_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.spiral_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 8)
    ax1.scatter(x1, y1)
    ax1.set_title('Spiral', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.spiral_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 8, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Spiral', fontweight='bold')
    ax2.axis('off')

    # Uncorrelated Bernoulli Simulation
    returns = sims.ubern_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.ubern_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 9)
    ax1.scatter(x1, y1)
    ax1.set_title('Uncorrelated Bernoulli', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.ubern_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 9, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Uncorrelated Bernoulli', fontweight='bold')
    ax2.axis('off')

    # Logarithmic Simulation
    returns_low_dim = sims.log_sim(num_samps, num_dim1)
    returns_high_dim = sims.log_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.log_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 10)
    ax1.scatter(x1, y1)
    ax1.set_title('Logarithmic', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.log_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 10, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2[:, 0]*y2[:, 1])
    ax2.set_title('Logarithmic', fontweight='bold')
    ax2.axis('off')

    # Nth Root Simulation
    returns_low_dim = sims.root_sim(num_samps, num_dim1)
    returns_high_dim = sims.root_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.root_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 11)
    ax1.scatter(x1, y1)
    ax1.set_title('Fourth Root', fontweight='bold')
    ax1.axis('off')

    x2, y2 = sims.root_sim(num_samps, 2, noise=0)
    ax2 = fig2.add_subplot(4, 5, 11, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], y2)
    ax2.set_title('Fourth Root', fontweight='bold')
    ax2.axis('off')

    # Sinusoidal Simulation (4*pi)
    returns_low_dim = sims.sin_sim(num_samps, num_dim1)
    returns_high_dim = sims.sin_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.sin_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 12)
    ax1.scatter(x1, y1)
    ax1.set_title('Sinusoidal (4\u03C0)', fontweight='bold')
    ax1.axis('off')

    # Sinusoidal Simulation (16*pi)
    returns_low_dim = sims.sin_sim(num_samps, num_dim1, period=16*np.pi)
    returns_high_dim = sims.sin_sim(
        num_samps, num_dim2, period=16*np.pi, indep=independent)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.sin_sim(num_samps, 1, noise=0, period=16*np.pi)
    ax1 = fig1.add_subplot(4, 5, 13)
    ax1.scatter(x1, y1)
    ax1.set_title('Sinusoidal (16\u03C0)', fontweight='bold')
    ax1.axis('off')

    # Square Simulation
    returns = sims.square_sim(num_samps, num_dim2, indep=independent)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.square_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 14)
    ax1.scatter(x1, y1)
    ax1.set_title('Square', fontweight='bold')
    ax1.axis('off')

    # Two Parabolas Simulation
    returns_low_dim = sims.two_parab_sim(num_samps, num_dim1)
    returns_high_dim = sims.two_parab_sim(num_samps, num_dim2)
    assert np.all(returns_low_dim[0].shape == (num_samps, num_dim1))
    assert np.all(returns_high_dim[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.two_parab_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 15)
    ax1.scatter(x1, y1)
    ax1.set_title('Two Parabolas', fontweight='bold')
    ax1.axis('off')

    # Circle Simulation
    returns = sims.circle_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.circle_sim(num_samps, 1, noise=0)
    ax1 = fig1.add_subplot(4, 5, 16)
    ax1.scatter(x1, y1)
    ax1.set_title('Circle', fontweight='bold')
    ax1.axis('off')

    # Ellipse Simulation
    returns = sims.circle_sim(num_samps, num_dim2, radius=5)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.circle_sim(num_samps, 1, noise=0, radius=5)
    ax1 = fig1.add_subplot(4, 5, 17)
    ax1.scatter(x1, y1)
    ax1.set_title('Ellipse', fontweight='bold')
    ax1.axis('off')

    # Diamond Simulation
    returns = sims.square_sim(
        num_samps, num_dim2, period=-np.pi/4, indep=independent)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.square_sim(num_samps, 1, noise=0, period=-np.pi/4)
    ax1 = fig1.add_subplot(4, 5, 18)
    ax1.scatter(x1, y1)
    ax1.set_title('Diamond', fontweight='bold')
    ax1.axis('off')

    # Multiplicative Noise Simulation
    returns = sims.multi_noise_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.multi_noise_sim(num_samps, 1)
    ax1 = fig1.add_subplot(4, 5, 19)
    ax1.scatter(x1, y1)
    ax1.set_title('Multiplicative Noise', fontweight='bold')
    ax1.axis('off')

    # Multimodal Independence Simulation
    returns = sims.multi_indep_sim(num_samps, num_dim2)
    assert np.all(returns[0].shape == (num_samps, num_dim2))

    x1, y1 = sims.multi_indep_sim(num_samps, 1)
    ax1 = fig1.add_subplot(4, 5, 20)
    ax1.scatter(x1, y1)
    ax1.set_title('Multimodal Independence', fontweight='bold')
    ax1.axis('off')


test_simulations()
