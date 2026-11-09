import numpy as np

def draw_bootstrap_sample(data, iterations=1000):
    """
    sample with replacement, data must be 1d
    """
    return np.random.choice(data, size=(iterations,len(data)))

def create_CI(data, CI=95.0, iterations=1000, sign=1):
    """
    can change sign of data in case want to do baseline-activation instead of activation-baseline, etc
    """
    
    sample_means = np.nanmean(draw_bootstrap_sample(sign*(data), iterations=iterations), axis=1)
    
    return np.percentile(sample_means, [100.0-CI, CI])

def create_CI_from_difference(data_1, data_2, CI=95.0, iterations=1000):
    """"
    generate a CI from the difference of 2 bootstrapped means, data2(stimulation) - data1(baseline) or reversed
    """
    
    sample_means_1 = np.nanmean(draw_bootstrap_sample(data_1, iterations=iterations), axis=1)
    sample_means_2 = np.nanmean(draw_bootstrap_sample(data_2, iterations=iterations), axis=1)
    
    # find deviation from baseline
    sample_means_diff = (sample_means_2 - sample_means_1)
    
    return np.percentile(sample_means_diff, [100.0-CI, CI])


def create_CI_timetrace(data, CI=95.0, iterations=1000):
    """Activation pointwise bootstrap, independently per timepoint."""
    CI_list_all = []
    for timepoint in range(np.shape(data)[1]):
        data_tpoint = data[:,timepoint]
        CI_list_tpoint = create_CI(data_tpoint, CI=CI, iterations=iterations)
        CI_list_all.append(CI_list_tpoint)
    return np.vstack(CI_list_all)



#stats helper

#helpers for permutation test
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    #concat 2 datasets
    concat_data = np.concatenate((x, y))
    #shuffle
    np.random.shuffle(concat_data)
    #reassign labels
    return concat_data[:len(x)], concat_data[len(x):]


def draw_perm_reps_diff_mean(x, y, size=10000000):
    """
    Generate array of permuation replicates.
    Difference x - y 
    """
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)
    return out

def permutation_test(x,y, size=1000000, vectorized=True, sided='two'):
    """
    Difference in means 
    x group 1 (control)
    y group 2 (experimental group)
    compute difference in means
    draw permutation replicates of difference in means 
    compute p value from # of occurances 
    sided=one sided ('greater', 'less') or 'two' sided
    """
    # Compute test statistic for original data set
    diff_mean = np.mean(x) - np.mean(y)

    # Draw replicates
    if vectorized:
        perm_reps = draw_perm_reps_diff_mean_vectorized(x, y, size=size)
    else:
        perm_reps = draw_perm_reps_diff_mean(x, y, size=size)

    #one sided 
    if sided=='greater':
    # Compute p-value (# of times perm_reps as big as unscrambled data / the number of reps)
        p_val = (np.sum(perm_reps >= diff_mean) + 1) / (len(perm_reps) + 1)
    elif sided=='less': #for cases where effect is <control
        p_val = (np.sum(perm_reps <= diff_mean) + 1) / (len(perm_reps) + 1)

    #2 sided 
    else:
        p_val = (np.sum(np.abs(perm_reps) >= np.abs(diff_mean))+1) / (len(perm_reps)+1)

    return p_val

def draw_perm_reps_diff_mean_vectorized(x, y, size=1000000):
    n_x = len(x)
    concat_data = np.concatenate((x, y))
    n_total = len(concat_data)

    # Generate `size` independent random permutations at once
    random_vals = np.random.random((size, n_total))
    #create randomized indices
    perm_indices = np.argsort(random_vals, axis=1)

    permuted = concat_data[perm_indices]  # shape (size, n_total)
    #first n vals, take mean, for every permutation
    x_perm_means = permuted[:, :n_x].mean(axis=1)
    y_perm_means = permuted[:, n_x:].mean(axis=1)

    return x_perm_means - y_perm_means
