def faster_mgc(data_matrix_X, data_matrix_Y, sub_samples=100, null_only=True, alpha=0.01):
    '''
    MGC test statistic computation and permutation test by fast subsampling.
    Note that trivial amount of noise is added to data_matrix_X and data_matrix_Y,
    to break possible ties in data for MGC.

    :param data_matrix_X: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type data_matrix_X: 2D numpy.array

    :param data_matrix_Y: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type data_matrix_Y: 2D numpy.array

    :param sub_samples: specifies the number of subsamples.
                        generally total_samples/sub_samples should be more than 4,
                        and sub_samples should be large than 30.
    :type sub_samples: int

    :param null_only: specifies if subsampling is to be used for estimating the null only OR
                      to compute the observed statistic as well
        - True: uses subsampled statistics for estimating the null only and computes the observed statistic by full data,
                this runs in O(total_samples^2 + sub_samples * total_samples)
        - False: uses subsampled statistics for estimating the null and also computes the observed statistic by subsampling,
                 this runs in O(sub_samples*total_samples)
    :type null_only: boolean

    :param alpha: specifies the type 1 error level.
                  this is is used to derive the confidence interval and estimate required sample size to achieve power 1.
    :type alpha: float

    :return: a ``dict`` of results with the following keys:
        - :p_value: P-value of MGC
        - :test_statistic: the sample MGC statistic within [-1, 1]
        - :local_correlation_matrix: a 2D matrix of all local correlations within [-1,1]
        - :optimal_scale: the estimated optimal scale as an [x, y] pair.
        - :confidence_interval: a [1*2] matrix representing the confidence_interval
                                for the local correlation with 1-alpha confidence.
        - :required_size: the required estimated sample size to have power 1 at level alpha
    '''

    pass
