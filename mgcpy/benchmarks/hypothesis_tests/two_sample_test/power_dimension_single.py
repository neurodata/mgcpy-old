import numpy as np
import math
import scipy.io
import os
from scipy.stats import special_ortho_group


def transform_matrices(x, y, is_y_categorical=False):
    if not is_y_categorical:
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    else:
        u = x
        v = preprocessing.LabelEncoder().fit_transform(y) + 1
    
    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]
    
    return u, v

def make_rot(dim):
    rot_mat = np.zeros((dim, dim))
    theta = np.random.random() * 180
    rand = np.random.random()
    if rand < 0.5:
        theta = theta*(-1)
    angle = math.radians(theta)
    rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    for i in range(dim):
        rot_mat[i][i]=1
    rot_mat[0][0] = rot[0][0]
    rot_mat[dim-1][0] = rot[1][0]
    rot_mat[0][dim-1] = rot[0][1]
    rot_mat[dim-1][dim-1] = rot[1][1]
    return rot_mat

def power_given_data(independence_test, simulation_type, num_samples=100, num_dimensions=1, translate=0, theta=0, repeats=1000, alpha=.05, additional_params={}):
    # test statistics under the null, used to estimate the cutoff value under the null distribution
    test_stats_null = np.zeros(repeats)
    # test statistic under the alternative
    test_stats_alternative = np.zeros(repeats)
    p_values = np.zeros(repeats)
    # absolute path to the benchmark directory
    file_name_prefix = 'matlabsim_{}_dim{}'.format(
                simulation_type, num_dimensions)
    all_matrix_X = scipy.io.loadmat(file_name_prefix + '_X_noise.mat')['x_mat']
    all_matrix_Y = scipy.io.loadmat(file_name_prefix + '_Y_noise.mat')['y_mat']
    for rep in range(repeats):
        ori_X = all_matrix_X[:, :, rep]
        ori_Y = all_matrix_Y[:, :, rep]
        fir_X = []
        fir_Y = []
        for tad in range(num_samples): 
            fir_X.append(ori_X[tad][0])
            fir_Y.append(ori_Y[tad][0])
        first_X = np.asarray(fir_X)
        first_Y = np.asarray(fir_Y)
        min_X = np.amin(first_X)
        min_Y = np.amin(first_Y)
        max_X = np.amax(first_X-min_X)
        max_Y = np.amax(first_Y-min_Y)
        new_X = (first_X-min_X)*2/max_X - 1
        new_Y = (first_Y-min_Y)*2/max_Y - 1
        matrix_X = np.zeros((num_samples,num_dimensions))
        matrix_Y = np.zeros((num_samples,1))
        for spot in range(num_samples):
            matrix_X[spot][0] = new_X[spot]
            matrix_Y[spot][0] = new_Y[spot]
            for d in range(1,num_dimensions):
                matrix_X[spot][d]=ori_X[spot][d]
                #matrix_Y[spot][d]=ori_Y[spot][d]
        data_matrix = transform_matrices(matrix_X.T, matrix_Y.T)[0]
        sim_list = [2, 3, 8, 9, 11, 16]
        if simulation_type in sim_list:
            q = make_rot(num_dimensions+1)
        else: 
            rot = np.zeros((num_dimensions+1,1))
            for i in range(num_dimensions+1):
                add=0
                mat = np.zeros((num_dimensions+1,1))
                for j in range(num_dimensions+1):
                    a = np.random.normal()
                    mat[j] = a
                    add = add + a**2
                norm = np.sqrt(add)
                mat = mat/norm
                if i==0:
                    rot = mat
                else:
                    rot = np.concatenate((rot, mat), axis=1)
            q,r = np.linalg.qr(rot)
            if num_dimensions%2==1:
                q[0] = q[0]*(-1)
        r_matrix = np.dot(q, data_matrix)
        r_trans = r_matrix.T
        trans_X = np.zeros((num_samples,num_dimensions))
        trans_Y = np.zeros((num_samples,1))
        for dot in range(num_samples):
            trans_X[dot][0] = r_trans[dot][0]+translate
            trans_Y[dot][0] = r_trans[dot][num_dimensions]
            for dime in range(1,num_dimensions):
                trans_X[dot][dime]=r_trans[dot][dime]
        trans_mat = transform_matrices(trans_X.T, trans_Y.T)[0]
        mat_X, mat_Y = transform_matrices(data_matrix.T, trans_mat.T)
        # permutation test
        if additional_params and additional_params["is_fast"]:
            p_values[rep], _ = independence_test.p_value(mat_X, mat_Y, **additional_params)
        else:
            permuted_Y = np.random.permutation(mat_Y)
            test_stats_null[rep], _ = independence_test.test_statistic(
                mat_X, permuted_Y, **additional_params)
            test_stats_alternative[rep], _ = independence_test.test_statistic(
                mat_X, mat_Y, **additional_params)
        '''
        # if the test is pearson, use absolute value of the test statistic
        # so the more extreme test statistic is still in a one-sided interval
        if independence_test.get_name() == 'pearson':
            test_stats_null[rep] = abs(test_stats_null[rep])
            test_stats_alternative[rep] = abs(test_stats_alternative[rep])
        '''

    if additional_params and additional_params["is_fast"]:
        empirical_power = np.where(p_values <= alpha)[0].shape[0] / repeats
    else:
        # the cutoff is determined so that 1-alpha of the test statistics under the null distribution
        # is less than the cutoff
        cutoff = np.sort(test_stats_null)[math.ceil(repeats*(1-alpha))]

        # the proportion of test statistics under the alternative which is no less than the cutoff (in which case
        # the null is rejected) is the empirical power
        empirical_power = np.where(test_stats_alternative >= cutoff)[0].shape[0] / repeats

    return empirical_power