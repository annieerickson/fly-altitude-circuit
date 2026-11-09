"""Activation extraction and statistical calculations.
"""

import numpy as np
from statsmodels.stats.multitest import multipletests

from ... import stats
from ...signals import get_start_end_indices_stimVolts_df, interpolate_trace

EXTRACTION_KEYS = ("stimVolts", "pre_stim", "post_stim", "stim_duration", "fs",
                   "post_stim_flying_thresh")


def extraction_settings(panel):
    """Select only the original extraction arguments from paper configuration."""
    return {key: panel[key] for key in EXTRACTION_KEYS}


def driver_name(driver):
    return "EV control" if driver == "S000" else "66183 DNa10" if driver == "S443" else "1070 DNa10"


def calculate_trace_results(groups, panel):
    """pointwise grand means and bootstrap intervals in driver order."""
    results = []
    for group in groups:
        mean = np.nanmean(group["diff"], axis=0)
        time = np.linspace(-panel["pre_stim"], panel["stim_duration"] + panel["post_stim"], len(mean))
        interval = None
        if panel["include_bs_CI"]:
            interval = stats.create_CI_timetrace(np.vstack(group["diff"]),
                                               iterations=panel["bootstrap_iterations"], CI=panel["ci"])
        results.append(dict(driver=group["driver"], mean=mean, time=time, ci=interval))
    return results


def calculate_response_results(groups, panel):
    """Scalar responses, CIs and three Holm-corrected permutations.
    """
    if len(groups) != 3:
        raise ValueError("The paper activation comparison requires exactly three driver groups.")
    results = []
    for group in groups:
        values = np.nanmean(group["diff"], axis=1)
        mean = np.nanmean(values)
        jitter = np.random.uniform(-panel["jitter"], panel["jitter"], len(values))
        interval = stats.create_CI(values, iterations=panel["bootstrap_iterations"], CI=panel["ci"])
        results.append(dict(driver=group["driver"], label=driver_name(group["driver"]),
                            values=values, mean=mean, jitter=jitter, ci=interval))
    
    args = [list(group["values"]) for group in results]
    pairs = panel["comparisons"]
    pvalues = [stats.permutation_test(args[x], args[y], size=panel["permutation_iterations"])
               for x, y in pairs]
    reject, corrected, _, _ = multipletests(pvalues, method=panel["correction"])
    return dict(groups=results, comparisons=pairs, pvalues=np.asarray(pvalues),
                corrected_pvalues=corrected, reject=reject)





def extract_trace_data(df, *, stimVolts, pre_stim, post_stim, stim_duration,
                       fs, post_stim_flying_thresh):
    """Extract L+R traces and baseline-subtracted responses"""
    trial_duration = pre_stim + stim_duration + post_stim
    frames = int(fs*trial_duration)
    frames_pre = int(fs*pre_stim)
    df_subset = df[df['trial_type']=='ChRT']
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    volt = stimVolts[0]
    groups = []
    for driver in drivers:
        flies = df_gb.xs(driver).fly_id.unique()
        all_fly_data = []
        all_fly_data_diff = []
        fly_n = 0
        pattern_n = 0
        included_fly_ids = []
        presentations = []
        for fly in flies:
            fly_data = []
            fly_data_diff = []
            curr_df = df_gb.xs((driver, fly))

            start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.25) 
            if len(start_times)!=0 and len(end_times)!=0:
                for start, stop in zip(start_times, end_times):
                    flight_vec = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop+post_stim_flying_thresh)]['flying_bool'] #True= not flying 
                    #if there are no flight pauses in this period then save data
                    if not (flight_vec==True).any():
                        presentations.append((fly, start, stop))
                        LW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop+post_stim)]['LWF']
                        RW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop+post_stim)]['RWF']

                        t_int = np.linspace(-pre_stim, stim_duration+post_stim,frames)
                        W_t = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop+post_stim)]['timeVec'] - start #timetrace zeroed at start
                        LW_int = interpolate_trace(W_t, LW, t_int)
                        LW_int_b = np.nanmean(LW_int[:frames_pre])
                        RW_int = interpolate_trace(W_t, RW, t_int)
                        RW_int_b = np.nanmean(RW_int[:frames_pre])
                        LpR_b = (LW_int_b + RW_int_b)
                        #add L+Rwing and baseline subtract 

                        #differences
                        LpR_diff = (RW_int+LW_int) - LpR_b
                        fly_data_diff.append(LpR_diff) 
                        #differences
                        LpR = (RW_int+LW_int)
                        fly_data.append(LpR) #L+R per trial/fly

            #mean of fly trials 
            if len(fly_data)!=0:
                fly_mean = np.nanmean(fly_data, axis=0)
                all_fly_data.append(fly_mean)

                fly_mean_diff = np.nanmean(fly_data_diff, axis=0)
                all_fly_data_diff.append(fly_mean_diff)

                fly_n+=1
                included_fly_ids.append(fly)
                pattern_n+=len(fly_data)
        groups.append(dict(driver=driver, fly_ids=included_fly_ids,
                           raw=all_fly_data, diff=all_fly_data_diff,
                           fly_n=fly_n, pattern_n=pattern_n,
                           presentations=presentations))
    return groups


def extract_response_data(df, *, stimVolts, pre_stim, post_stim, stim_duration,
                       fs, post_stim_flying_thresh):
    """Extract L+R stimulus responses with baseline subtraction."""
    trial_duration = pre_stim + stim_duration
    frames = int(fs*trial_duration)
    frames_pre = int(fs*pre_stim)
    df_subset = df[df['trial_type']=='ChRT']
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    volt = stimVolts[0]
    groups = []
    for driver in drivers:
        flies = df_gb.xs(driver).fly_id.unique()
        all_fly_data = []
        all_fly_data_diff = []
        fly_n = 0
        pattern_n = 0
        included_fly_ids = []
        presentations = []
        for fly in flies:
            fly_data = []
            fly_data_diff = []
            curr_df = df_gb.xs((driver, fly))

            start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.25)
            if len(start_times)!=0 and len(end_times)!=0:
                for start, stop in zip(start_times, end_times):
                    flight_vec = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop+post_stim_flying_thresh)]['flying_bool'] #True= not flying 
                    #if there are no flight pauses in this period then save data (still exclude if flight stop near stim period)
                    if not (flight_vec==True).any():
                        presentations.append((fly, start, stop))

                        #only care about during stimulus change (and pre period for baseline calc)
                        LW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop)]['LWF']
                        RW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop)]['RWF']

                        t_int = np.linspace(-pre_stim, stim_duration,frames)
                        W_t = curr_df.loc[(curr_df['timeVec']>=start-pre_stim) & (curr_df['timeVec']<stop)]['timeVec'] - start #timetrace zeroed at start
                        LW_int = interpolate_trace(W_t, LW, t_int)
                        LW_int_b = np.nanmean(LW_int[:frames_pre])
                        RW_int = interpolate_trace(W_t, RW, t_int)
                        RW_int_b = np.nanmean(RW_int[:frames_pre])
                        LpR_b = (LW_int_b + RW_int_b)
                        #add L+Rwing and baseline subtract 

                        #differences
                        LpR_diff = (RW_int+LW_int) - LpR_b #only want during stimulus period
                        fly_data_diff.append(LpR_diff[frames_pre:]) 
                        #differences
                        LpR = (RW_int+LW_int)
                        fly_data.append(LpR[frames_pre:]) #L+R per trial/fly

            #mean of fly trials 
            if len(fly_data)!=0:
                fly_mean = np.nanmean(fly_data, axis=0)
                all_fly_data.append(fly_mean)

                fly_mean_diff = np.nanmean(fly_data_diff, axis=0)
                all_fly_data_diff.append(fly_mean_diff)

                fly_n+=1
                included_fly_ids.append(fly)
                pattern_n+=len(fly_data)
        groups.append(dict(driver=driver, fly_ids=included_fly_ids,
                           raw=all_fly_data, diff=all_fly_data_diff,
                           fly_n=fly_n, pattern_n=pattern_n,
                           presentations=presentations))
    return groups

