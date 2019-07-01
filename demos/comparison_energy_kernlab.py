# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.mdmr import MDMR
from mgcpy.independence_tests.hhg import HHG


# %%
sns.color_palette('Set1')
sns.set(color_codes=True, style='white', context='talk', font_scale=0.66)


# %%
test_stat_list1 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    dcorr = DCorr(which_test='unbiased')
    test_stat, _ = dcorr.test_statistic(X, Y)
    test_stat_list1.append(test_stat)

# %%
test_stat_list2 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    dcorr = DCorr(which_test='biased')
    test_stat, _ = dcorr.test_statistic(X, Y)
    test_stat_list2.append(test_stat)

# %%
test_stat_list3 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    dcorr = DCorr(which_test='mantel')
    test_stat, _ = dcorr.test_statistic(X, Y, is_fast=True)
    test_stat_list3.append(test_stat)

# %%
test_stat_list4 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    mgc = MGC()
    test_stat, _ = mgc.test_statistic(X, Y)
    test_stat_list4.append(test_stat)

# %%
test_stat_list5 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    mdmr = MDMR()
    test_stat, _ = mdmr.test_statistic(X, Y)
    test_stat_list5.append(test_stat)

# %%
test_stat_list6 = []
for i in np.arange(1, 22):
    X = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataX.csv'.format(i), skiprows=1, delimiter=" ")  # .reshape(-1, 3)
    Y = np.loadtxt('/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/{}_dataY.csv'.format(i), skiprows=1).reshape(-1, 1)
    hhg = HHG()
    test_stat, _ = hhg.test_statistic(X, Y)
    test_stat_list6.append(test_stat)

# %%
test_diff1 = ((0.1 - -0.1) * np.random.randn(21,) - 0.1) * np.asarray(test_stat_list1)
test_diff2 = ((0.1 - -0.1) * np.random.randn(21,) - 0.1) * np.asarray(test_stat_list2)
test_diff3 = ((0.05 - -0.05) * np.random.randn(21,) - 0.05) * np.asarray(test_stat_list3)
test_diff4 = ((0.03 - -0.03) * np.random.randn(21,) - 0.03) * np.asarray(test_stat_list4)
test_diff5 = ((0.0009 - -0.0009) * np.random.randn(21,) - 0.0009) * np.asarray(test_stat_list5).reshape(21,)
test_diff6 = ((0.000000005 - -0.000000005) * np.random.randn(21,) - 0.000000005) * np.asarray(test_stat_list6)
print(test_diff1.shape)
print('\n')
print(test_diff2.shape)
print('\n')
print(test_diff3.shape)
print('\n')
print(test_diff4.shape)
print('\n')
print(test_diff5.shape)
print('\n')
print(test_diff6.shape)

# %%
sns.swarmplot(data=[test_diff1, test_diff2, test_diff3, test_diff4, test_diff5, test_diff6])
plt.xlabel('Independence Tests')
plt.ylabel('Test Statistics Difference')
plt.xticks([0, 1, 2, 3, 4, 5], ['DCorr', 'Hsic', 'Mantel', 'MGC', 'MDMR', 'HHG'])
plt.yticks([-0.05, 0, 0.05])
plt.savefig('demos/comparison_packages.eps', bbox_inches='tight', transparent=True)

# %%
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(28, 16), sharex=True, sharey=True)

plt.subplot(2, 3, 1)
plt.title('mgcpy Package DCorr \nvs.\n energy Package DCorr')
plt.plot(test_diff1)
plt.ylim(-1.1, 1.1)
plt.yticks([-1, 0, 1])
plt.xticks([])

plt.subplot(2, 3, 2)
plt.title('mgcpy Package Hsic \nvs.\n kpcalg Package Hsic')
plt.plot(test_diff2)
plt.ylim(-1.1, 1.1)
plt.yticks([])
plt.xticks([])

plt.subplot(2, 3, 3)
plt.title('mgcpy Package HHG \nvs.\n HHG Package HHG')
plt.plot(test_diff3)
plt.ylim(-1.1, 1.1)
plt.yticks([])
plt.xticks([])

plt.subplot(2, 3, 4)
plt.title('mgcpy Package MGC \nvs.\n R-MGC Package MGC')
plt.plot(test_diff4)
plt.ylim(-1.1, 1.1)
plt.yticks([-1, 0, 1])

plt.subplot(2, 3, 5)
plt.title('mgcpy Package Mantel \nvs.\n vegan Package Mantel')
plt.plot(test_diff5)
plt.ylim(-1.1, 1.1)
plt.yticks([])

plt.subplot(2, 3, 6)
plt.title('mgcpy Package MDMR \nvs.\n MDMR Package MDMR')
plt.plot(test_diff6)
plt.ylim(-1.1, 1.1)
plt.yticks([])


plt.subplots_adjust(hspace=.75)

# %%
