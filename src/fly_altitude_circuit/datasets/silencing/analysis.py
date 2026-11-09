"""Silencing extraction paths and statistics."""

import numpy as np
from ... import stats
from ...signals import get_start_end_indices_stimulus_df, get_start_end_indices_stimVolts_df, interpolate_trace


def validate_paper_settings(panel):
    if panel["kind"].startswith("ux") and list(panel["patterns"]) != [11.0]:
        raise ValueError("Only single-pattern 11.0 UX configuration is suported.")
    if panel["kind"].startswith("gtacr") and panel["exp_name"] != "GTCL":
        raise ValueError("Only GTCL condition is supported.")


def calculate_results(extracted, panel):
    kind = panel["kind"]
    groups = extracted["groups"]
    if {group["driver"] for group in groups} != {"S000", "S442"} or len(groups) != 2:
        raise ValueError("The silencing comparisons require the S000 and S442 groups.")
    results = []
    for group in groups:
        if kind.endswith("trace"):
            mean_raw = np.nanmean(group["raw"], axis=0)
            time = np.linspace(-panel["pre_stim"], panel["stim_duration"] + panel["post_stim"], len(mean_raw))
            raw_ci = stats.create_CI_timetrace(np.vstack(group["raw"]), iterations=panel["bootstrap_iterations"], CI=panel["ci"])
            mean_diff = np.nanmean(group["diff"], axis=0)
            diff_ci = stats.create_CI_timetrace(np.vstack(group["diff"]), iterations=panel["bootstrap_iterations"], CI=panel["ci"])
            results.append(dict(driver=group["driver"], time=time, raw=mean_raw, diff=mean_diff,
                                raw_ci=raw_ci, diff_ci=diff_ci))
        else:
            values = group["values"]
            mean = np.nanmean(values, axis=0)
            interval = stats.create_CI(values, iterations=panel["bootstrap_iterations"], CI=panel["ci"])
            jitter = np.random.uniform(-panel["jitter"], panel["jitter"], len(values))
            results.append(dict(driver=group["driver"], values=values, mean=mean, ci=interval, jitter=jitter))
    output = dict(groups=results)
    by_driver = {group["driver"]: group["values"] for group in groups}
    if kind in ("ux_trace", "ux_baseline", "ux_response"):
        if kind == "ux_baseline":
            control, experimental = extracted["control"], extracted["experimental"]
            difference = np.nanmean(experimental) - np.nanmean(control)
            interval = stats.create_CI_from_difference(control, experimental)
            relative_interval = [(x / np.nanmean(control)) * 100 for x in interval]
            relative = (difference / np.nanmean(control)) * 100
            unit = "percent"
        else:
            control, experimental = by_driver["S000"], by_driver["S442"]
            difference = np.nanmean(control) - np.nanmean(experimental)
            interval = stats.create_CI_from_difference(experimental, control)
            relative_interval = [x / np.nanmean(control) for x in interval]
            relative = difference / np.nanmean(control)
            unit = "fraction"
        output["effect"] = dict(difference=difference, ci=interval, relative=relative,
                                relative_ci=relative_interval, relative_unit=unit)
    if not kind.endswith("trace"):
        args = [list(extracted["control"]), list(extracted["experimental"])] if kind == "ux_baseline" else [list(group["values"]) for group in groups]
        output["pvalue"] = stats.permutation_test(args[0], args[1], size=panel["permutation_iterations"])
    return output


def extract_ux_traces(df, panel):
    post_stim = panel['post_stim']
    stim_duration = panel['stim_duration']
    fs = panel['fs']
    pre_stim = panel['pre_stim']
    post_stim_flying_thresh = panel['post_stim_flying_thresh']
    patdict = panel["patterns"]
    trial_duration = pre_stim + stim_duration + post_stim
    frames = int(fs * trial_duration)
    frames_pre = int(fs * pre_stim)
    frames_stim = int(fs * stim_duration)
    df_subset = df[df['trial_type'] != 'upRT']
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    groups = []
    for pat_id, pat in enumerate(patdict.keys()):
        for driver in drivers:
            flies = df_gb.xs(driver).fly_id.unique()
            all_fly_data = []
            all_fly_data_diff = []
            all_fly_mean_changes = []
            fly_n = 0
            pattern_n = 0
            included_fly_ids = []
            presentations = []
            for fly in flies:
                fly_data = []
                fly_data_diff = []
                diff_change_per_fly = []
                curr_df = df_gb.xs((driver, fly))
                if pat in [3.0,7.0]: #need to subtract-1 from 2,6 trans down half volts
                    adjust_volt=True
                else:
                    adjust_volt=False 

                start_times, end_times = get_start_end_indices_stimulus_df(curr_df, stimulus_volt=pat, range_volts=0.25, half_volt=True, adjust_volt=adjust_volt)
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
                            fly_data_diff.append(LpR_diff) #L+R per trial/fly
                            #differences
                            LpR = (RW_int+LW_int)
                            fly_data.append(LpR)
                            #mean change during full stim period per trial 
                            fly_diff_change = np.nanmean(LpR_diff[frames_pre:frames_pre+frames_stim]) 
                            diff_change_per_fly.append(fly_diff_change) #L+R per trial/fly


                #mean of fly trials 
                if len(fly_data)!=0:
                    fly_mean = np.nanmean(fly_data, axis=0)
                    all_fly_data.append(fly_mean)
                    included_fly_ids.append(fly)

                    fly_mean_diff = np.nanmean(fly_data_diff, axis=0)
                    all_fly_data_diff.append(fly_mean_diff)
                    #to calc reduction in response
                    np.nanmean(fly_data_diff)

                    #get mean change across flies 
                    fly_change = np.nanmean(diff_change_per_fly)
                    all_fly_mean_changes.append(fly_change)

                    fly_n+=1
                    pattern_n+=len(fly_data)
            groups.append(dict(driver=driver, values=all_fly_mean_changes, fly_ids=included_fly_ids, fly_n=fly_n, pattern_n=pattern_n, presentations=presentations, raw=all_fly_data, diff=all_fly_data_diff))
    return dict(groups=groups)


def extract_ux_baseline(df, panel):
    post_stim = panel['post_stim']
    stim_duration = panel['stim_duration']
    fs = panel['fs']
    pre_stim_baseline_period = panel['pre_stim_baseline_period']
    patdict = panel["patterns"]
    trial_duration = pre_stim_baseline_period + stim_duration + post_stim
    frames = fs * trial_duration
    frames_pre = fs * pre_stim_baseline_period
    df_subset = df[df['trial_type'] != 'upRT']
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    groups = []
    all_pat_fly_data_s00 = []
    all_pat_fly_data_s442 = []
    for driver in drivers:
        flies = df_gb.xs(driver).fly_id.unique()
        all_fly_data = []
        fly_n = 0
        pattern_n = 0
        included_fly_ids = []
        presentations = []
        for fly in flies:

            for pat_id, pat in enumerate(patdict.keys()): 
                fly_data = []
                curr_df = df_gb.xs((driver, fly))
                if pat in [3.0,7.0]: #need to subtract-1 from 2,6 trans down half volts
                    adjust_volt=True
                else:
                    adjust_volt=False 

                start_times, end_times = get_start_end_indices_stimulus_df(curr_df, stimulus_volt=pat, range_volts=0.25, half_volt=True, adjust_volt=adjust_volt)
                if len(start_times)!=0 and len(end_times)!=0:
                    for start, stop in zip(start_times, end_times):
                        flight_vec = curr_df.loc[(curr_df['timeVec']>=start-pre_stim_baseline_period) & (curr_df['timeVec']<stop+post_stim)]['flying_bool'] #True= not flying 
                        #if there are no flight pauses in this period then save data
                        if not (flight_vec==True).any():
                            presentations.append((fly, start, stop))
                            LW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim_baseline_period) & (curr_df['timeVec']<stop+post_stim)]['LWF']
                            RW = curr_df.loc[(curr_df['timeVec']>=start-pre_stim_baseline_period) & (curr_df['timeVec']<stop+post_stim)]['RWF']

                            t_int = np.linspace(-pre_stim_baseline_period, stim_duration+post_stim,frames)
                            W_t = curr_df.loc[(curr_df['timeVec']>=start-pre_stim_baseline_period) & (curr_df['timeVec']<stop+post_stim)]['timeVec'] - start #timetrace zeroed at start
                            LW_int = interpolate_trace(W_t, LW, t_int)
                            RW_int = interpolate_trace(W_t, RW, t_int)

                            #only get the baseline periods
                            LW_int_b = np.nanmean(LW_int[:frames_pre])
                            RW_int_b = np.nanmean(RW_int[:frames_pre])

                            LpR_b = (LW_int_b + RW_int_b)
                            fly_data.append(LpR_b) #baseline per trial per fly

                            #add L+Rwing and baseline subtract 
                            # LpR = (RW_int+LW_int) - LpR_b


            #mean of fly trials per fly
            if len(fly_data)!=0:
                fly_mean = np.nanmean(fly_data, axis=0)
                all_fly_data.append(fly_mean)
                included_fly_ids.append(fly)
                fly_n += 1
                pattern_n += len(fly_data)
                if driver=='S000':
                    all_pat_fly_data_s00.append(fly_mean)
                else:
                    all_pat_fly_data_s442.append(fly_mean)
        groups.append(dict(driver=driver, values=all_fly_data, fly_ids=included_fly_ids, fly_n=fly_n, pattern_n=pattern_n, presentations=presentations))
    return dict(groups=groups, control=all_pat_fly_data_s00, experimental=all_pat_fly_data_s442)


def extract_ux_response(df, panel):
    post_stim = panel['post_stim']
    stim_duration = panel['stim_duration']
    fs = panel['fs']
    pre_stim = panel['pre_stim']
    post_stim_flying_thresh = panel['post_stim_flying_thresh']
    patdict = panel["patterns"]
    trial_duration = pre_stim + stim_duration + post_stim
    frames = int(fs * trial_duration)
    frames_pre = int(fs * pre_stim)
    frames_stim = int(fs * stim_duration)
    df_subset = df[df['trial_type'] != 'upRT']
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    groups = []
    for pat_id, pat in enumerate(patdict.keys()):
        for driver in drivers:
            flies = df_gb.xs(driver).fly_id.unique()
            all_fly_data = []
            all_fly_data_diff = []
            all_fly_mean_changes = []
            fly_n = 0
            pattern_n = 0
            included_fly_ids = []
            presentations = []
            for fly in flies:
                fly_data = []
                fly_data_diff = []
                diff_change_per_fly = []
                curr_df = df_gb.xs((driver, fly))
                if pat in [3.0,7.0]: #need to subtract-1 from 2,6 trans down half volts
                    adjust_volt=True
                else:
                    adjust_volt=False 

                start_times, end_times = get_start_end_indices_stimulus_df(curr_df, stimulus_volt=pat, range_volts=0.25, half_volt=True, adjust_volt=adjust_volt)
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
                            fly_data_diff.append(LpR_diff) #L+R per trial/fly
                            #differences
                            LpR = (RW_int+LW_int)
                            fly_data.append(LpR)
                            #mean change during full stim period per trial 
                            fly_diff_change = np.nanmean(LpR_diff[frames_pre:frames_pre+frames_stim]) 
                            diff_change_per_fly.append(fly_diff_change) #L+R per trial/fly


                #mean of fly trials 
                if len(fly_data)!=0:
                    fly_mean = np.nanmean(fly_data, axis=0)
                    all_fly_data.append(fly_mean)
                    included_fly_ids.append(fly)

                    fly_mean_diff = np.nanmean(fly_data_diff, axis=0)
                    all_fly_data_diff.append(fly_mean_diff)
                    #to calc reduction in response
                    np.nanmean(fly_data_diff)

                    #get mean change across flies 
                    fly_change = np.nanmean(diff_change_per_fly)
                    all_fly_mean_changes.append(fly_change)

                    fly_n+=1
                    pattern_n+=len(fly_data)
            groups.append(dict(driver=driver, values=all_fly_mean_changes, fly_ids=included_fly_ids, fly_n=fly_n, pattern_n=pattern_n, presentations=presentations))
    return dict(groups=groups)


def extract_gtacr_traces(df, panel):
    post_stim = panel['post_stim']
    stim_duration = panel['stim_duration']
    fs = panel['fs']
    pre_stim = panel['pre_stim']
    post_stim_flying_thresh = panel['post_stim_flying_thresh']
    volt = panel['volt']
    exp_name = panel['exp_name']
    trial_duration = pre_stim + stim_duration + post_stim
    frames = int(fs * trial_duration)
    frames_pre = int(fs * pre_stim)
    df_subset = df[df['trial_type'] == exp_name]
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
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

            if (curr_df.driver[0]=='S000') and (curr_df.date[0]=='250218') and (curr_df.trial_num[0]=='004') and (exp_name=='GTDk'): #wings not tracked right in this file
                continue

            if (exp_name=='GTCL') or (exp_name=='GTDk'): #only light on
                start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.25)
            else: #light on and visual pattern
                start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.05) 

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
                        fly_data_diff.append(LpR_diff) #L+R per trial/fly, exclude last point since it is not interpolated
                        #differences
                        LpR = (RW_int+LW_int)
                        fly_data.append(LpR) #L+R per trial/fly


            #mean of fly trials 
            if len(fly_data)!=0:
                fly_mean = np.nanmean(fly_data, axis=0)
                all_fly_data.append(fly_mean)
                included_fly_ids.append(fly)

                fly_mean_diff = np.nanmean(fly_data_diff, axis=0)
                all_fly_data_diff.append(fly_mean_diff)

                fly_n+=1
                pattern_n+=len(fly_data)
        groups.append(dict(driver=driver, values=all_fly_data, fly_ids=included_fly_ids, fly_n=fly_n, pattern_n=pattern_n, presentations=presentations, raw=all_fly_data, diff=all_fly_data_diff))
    return dict(groups=groups)


def extract_gtacr_response(df, panel):
    post_stim = panel['post_stim']
    stim_duration = panel['stim_duration']
    fs = panel['fs']
    pre_stim = panel['pre_stim']
    post_stim_flying_thresh = panel['post_stim_flying_thresh']
    volt = panel['volt']
    exp_name = panel['exp_name']
    mean_period = panel['mean_period']
    trial_duration = pre_stim + stim_duration + post_stim
    frames = int(fs * trial_duration)
    frames_pre = int(fs * pre_stim)
    frames_stim = int(fs * stim_duration)
    frames_for_mean = int(fs * mean_period)
    df_subset = df[df['trial_type'] == exp_name]
    df_gb = df_subset.groupby(['driver', 'fly_id'], group_keys=True).apply(lambda x: x)
    drivers = df_subset.driver.unique()
    groups = []
    for driver in drivers:
        flies = df_gb.xs(driver).fly_id.unique()
        all_fly_data = []
        fly_n = 0
        pattern_n = 0
        included_fly_ids = []
        presentations = []
        for fly in flies:
            fly_data = []
            # fly_data_diff = []
            # fly_data_abs_baseline = []
            # fly_data_diff_baseline= [] 
            # fly_data_abs_stim = []
            # fly_data_diff_stim = []

            curr_df = df_gb.xs((driver, fly))

            if (curr_df.driver[0]=='S000') and (curr_df.date[0]=='250218') and (curr_df.trial_num[0]=='004') and (exp_name=='GTDk'): #wings not tracked right in this file
                continue

            if (exp_name=='GTCL') or (exp_name=='GTDk'): #only light on
                start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.25)
            else: #light on and visual pattern
                start_times, end_times = get_start_end_indices_stimVolts_df(curr_df, volt, range_volts=0.05) 

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


                        # #differences
                        LpR = (RW_int+LW_int)

                        #stim
                        if np.sign(frames_for_mean)==-1: #last 1s
                            abs_stim = np.nanmean(LpR[frames_pre + frames_stim + frames_for_mean:frames_pre + frames_stim])

                        else: #first 1s etc
                            abs_stim = np.nanmean(LpR[frames_pre :frames_pre + frames_for_mean])

                        change = abs_stim - LpR_b
                        fly_data.append(change)


            #mean of fly trials 
            if len(fly_data)!=0:
                fly_mean = np.nanmean(fly_data, axis=0)
                all_fly_data.append(fly_mean)
                included_fly_ids.append(fly)


                fly_n+=1
                pattern_n+=len(fly_data)
        groups.append(dict(driver=driver, values=all_fly_data, fly_ids=included_fly_ids, fly_n=fly_n, pattern_n=pattern_n, presentations=presentations))
    return dict(groups=groups)

