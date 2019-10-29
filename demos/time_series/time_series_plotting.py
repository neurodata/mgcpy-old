# Functions for plotting univraite time series and power curves for other processes.
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

def _compute_power(test, X_full, Y_full, num_sims, alpha, n):
    """
    Helper method estimate power of a test on a given simulation.

    :param test: Test to profile, either DCorrX or MGCX.
    :type test: TimeSeriesIndependenceTest

    :param X_full: An ``[n*num_sims]`` data matrix where ``n`` is the highest sample size.
    :type X_full: 2D ``numpy.array``

    :param Y_full: An ``[n*num_sims]`` data matrix where ``n`` is the highest sample size.
    :type Y_full: 2D ``numpy.array``

    :param num_sims: number of simulation at each sample size.
    :type num_sims: integer

    :param alpha: significance level.
    :type alpha: float

    :param n: sample size.
    :type n: integer

    :return: returns the estimated power.
    :rtype: float
    """
    num_rejects = 0.0

    def worker(s):
        X = X_full[range(n), s]
        Y = Y_full[range(n), s]

        p_value, _ = test['object'].p_value(X, Y, is_fast = test['is_fast'], subsample_size = test['subsample_size'])
        if p_value <= alpha:
            return 1

        return 0

    num_rejects = np.sum(Parallel(n_jobs=-2, verbose=10)(delayed(worker)(s) for s in range(num_sims)))

    return num_rejects / num_sims

def _plot_power(tests, sample_sizes, alpha, process):
    """
    Helper method to generate power curves for time series.

    :param tests: An array-like object containing TimeSeriesIndependenceTest objects.
    :type tests: 1-D array-like

    :param sample_sizes: range of sample sizes for which to estimate power.
    :type sample_sizes: 1-D array-like

    :param alpha: significance level.
    :type alpha: float

    :param process: A TimeSeriesProcess object for which to profile the test.
    :type process: TimeSeriesProcess
    """
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.title(process.name)
    plt.xlabel("n")
    plt.ylabel("Rejection Probability")
    plt.ylim((-0.05, 1.05))

    for test in tests:
        plt.plot(sample_sizes, test['powers'], linestyle = '-', color = test['color'])
    ax.legend([test['name'] for test in tests], loc = 'upper left', prop={'size': 12})

    ax.axhline(y = alpha, color = 'black', linestyle = '--')
    plt.show()

def plot_1d_ts(X, Y, title, xlab = "X_t", ylab = "Y_t"):
    """
    Method to plot univariate time series on the same figure.

    :param X: An ``[n*1]`` data matrix where ``n`` is the sample size.
    :type X: 2D ``numpy.array``

    :param Y: An ``[n*1]`` data matrix where ``n`` is the sample size.
    :type Y: 2D ``numpy.array``

    :param title: Plot title.
    :type title: string

    :param xlab: x-axis label.
    :type xlab: string

    :param ylab: y-axis label.
    :type ylab: string
    """
    n = X.shape[0]
    t = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7.5))
    fig.suptitle(title)
    plt.rcParams.update({'font.size': 15})

    ax1.plot(t, X)
    ax1.plot(t, Y)
    ax1.legend(['X_t', 'Y_t'], loc = 'upper left', prop={'size': 12})
    ax1.set_xlabel("t")

    ax2.scatter(X,Y, color="black")
    ax2.set_ylabel(ylab)
    ax2.set_xlabel(xlab)

# Power computation functions.

def power_curve(tests, process, num_sims, alpha, sample_size, verbose = False):
    """
    Method to generate power curves for time series.

    :param tests: An array-like object containing TimeSeriesIndependenceTest objects.
    :type tests: 1-D array-like

    :param process: A TimeSeriesProcess object for which to profile the test.
    :type process: TimeSeriesProcess

    :param num_sims: number of simulation at each sample size.
    :type num_sims: integer

    :param alpha: significance level.
    :type alpha: float

    :param verbose: whether to display output.
    :type verbose: boolean

    :param sample_sizes: range of sample sizes for which to estimate power.
    :type sample_sizes: 1-D array-like
    """
    # Store simulate processes.
    n_full = sample_sizes[len(sample_sizes) - 1]
    X_full = np.zeros((n_full, num_sims))
    Y_full = np.zeros((n_full, num_sims))
    for s in range(num_sims):
        X_full[:, s], Y_full[:, s] = process.simulate(n_full)

    for test in tests:
        powers = np.zeros(len(sample_sizes))
        for i in range(len(sample_sizes)):
            n = sample_sizes[i]
            if verbose: print("Estimating power at sample size: %d" % n)
            powers[i] = _compute_power(test, X_full, Y_full, num_sims, alpha, n)
        test['powers'] = powers

    # Display.
    _plot_power(tests, sample_sizes, alpha, process)

def plot_optimal_lags(optimal_lags, process, test_name, color, true_correlations = None, savefig = True):
    """
    Visualize distribution of optimal lag estimates. If the time series process is linear, then plot true cross correlations.

    :param optimal_lags: An array-like object containing optimal lag estimates from each simulation.
    :type optimal_lags: 1-D array-like

    :param process: A TimeSeriesProcess object for which to profile the test.
    :type process: TimeSeriesProcess
    
    :param test_name: String for the filename, either 'mgcx' or 'dcorrx'.
    :type test_name: string

    :param color: Color of histogram bars. Typically red for 'mgcx' and blue for 'dcorrx'.
    :type color: string

    :param true_correlations: Array of size ``max_lag`` + 1, containing the cross correlation of ``X_{t}`` and ``Y_{t-j}`` for each ``j``.
    :type true_correlations: 1-D array-like

    :param savefig: Whether to save the figure. Defaults to True.
    :type savefig: boolean
    """
    plt.rcParams.update({'font.size': 14})
    
    plt.xlabel('Lag j')
    if true_correlations is not None:
        # True correlations of X_{t} and Y_{t-j}at various lags.
        plt.ylabel("Corr(X(t), Y(t-j)) / Freq. of Optimal Lag Estimates")

        j = range(true_correlations.shape[0])
        markerline, stemlines, baseline = plt.stem(j, true_correlations, '-k')
        plt.setp(baseline, 'color', 'k', 'linewidth', 1)
        plt.setp(markerline, 'markerfacecolor', 'k')
    else:
        plt.ylabel("Freq. of Optimal Lag Estimates")
    
    # Optimal lab predictions.
    weights = np.ones_like(optimal_lags)/float(len(optimal_lags))
    plt.hist(optimal_lags, 
             bins = np.arange(len(true_correlations))-0.5, 
             weights = weights, 
             align = 'mid',
             edgecolor ='black',
             color = color)
    
    plt.title(process.title)
    if savefig:
        filename = "optimal_lags_%s_%s.pdf" % (process.filename, test_name)
        plt.savefig(filename)
    plt.show()


# def opt_lag_dist_stems(optimal_lags, true_correlations, title, color = 'red', savefig = True):
#     plt.rcParams.update({'font.size': 14})
    
#     # True correlations at various lags.
#     j = range(true_correlations.shape[0])
#     markerline, stemlines, baseline = plt.stem(j, true_correlations, '-k')
#     plt.setp(baseline, 'color', 'k', 'linewidth', 1)
#     plt.setp(markerline, 'markerfacecolor', 'k')
#     plt.xlabel('Lag j')
#     plt.ylabel("Corr(X(t), Y(t-j)) / Freq. of Optimal Lag Estimates")
    
#     # Optimal lab predictions.
#     weights = np.ones_like(optimal_lags)/float(len(optimal_lags))
#     plt.hist(optimal_lags, 
#              bins = np.arange(len(true_correlations))-0.5, 
#              weights = weights, 
#              align = 'mid',
#              edgecolor ='black',
#              color = color)
    
#     filename = "optimal_lag_dist_stems_%s.png" % format_filename(title)
#     if savefig:
#         plt.title(title)
#         plt.savefig(filename)
#     plt.show()

# def opt_lag_dist(optimal_lags, title, color = 'red', savefig = True):
#     plt.rcParams.update({'font.size': 14})
    
#     plt.xlabel('Lag j')
#     plt.ylabel("Freq. of Optimal Lag Estimates")
    
#     # Optimal lab predictions.
#     weights = np.ones_like(optimal_lags)/float(len(optimal_lags))
#     plt.hist(optimal_lags, 
#              bins = np.arange(len(true_correlations))-0.5, 
#              weights = weights, 
#              align = 'mid',
#              edgecolor ='black',
#              color = color)
    
#     filename = "optimal_lag_dist_%s.png" % format_filename(title)
#     if savefig:
#         plt.title(title)
#         plt.savefig(filename)
#     plt.show()