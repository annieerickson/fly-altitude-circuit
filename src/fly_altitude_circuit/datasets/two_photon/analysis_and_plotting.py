"""Two-photon workflows.
"""
from ... import config, plotting

import numpy as np
import pandas as pd
from scipy import signal, stats, odr
from collections import defaultdict
from matplotlib.patches import Rectangle
from ...signals import interpolate_trace
from ...stats import create_CI, create_CI_timetrace, permutation_test
from .io import load_combined, loom_path
import matplotlib.pyplot as plt

def z_score(arr):
    """
    return z score of arr
    """
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    z_score = (arr - mean) / std
    return z_score

def filtdFF(inArr, imFreqHz, n=4, wn=5):
    d, c = signal.butter(N=n, Wn=wn, btype='low', fs=imFreqHz)
    return signal.filtfilt(d, c, inArr)

def odr_target_function(B, X):
    """
    target function for odr to fit to
    """
    return B[0] * X + B[1]

def orthogonal_distance_regression(x, y, sx, sy, beta_estimate_0=1.0, beta_estimate_1=-0.5):
    """
    orthogonal distance regression 
    """
    linear = odr.Model(odr_target_function)
    data = odr.RealData(x, y, sx=sx, sy=sy)
    my_odr = odr.ODR(data, linear, beta0=[beta_estimate_0, beta_estimate_1])
    output = my_odr.run()
    return output

def get_start_end_indices_stimulus_df(stim_df, stimulus_volt, range_volts=0.25, half_volt=True):
    """
    return arrays of start and end indices of particular stimulus volts += range volts (not really indices- returns timeVec values)
    stim_df: dataframe of all phidget/behavioral/stimuli output (at imaging hz)
    stimulus_volt: value in pattern dict define above
    range_volts: upper and lower limits of acceptable variability in voltage signal (since separate 2.5 and 3 voltages, need 0.25 for no overlap)
    double_volt: if you expect the voltage to be 1/2 of the value in the pattern dictionary 
    for loom use smaller range_volts (0.075)
    """
    if half_volt == True:
        stim_volt = stimulus_volt * 0.5
    else:
        stim_volt = stimulus_volt
    lower_volt = stim_volt - range_volts
    upper_volt = stim_volt + range_volts
    indices = stim_df.loc[(stim_df['AuxPhidgAO1'] >= lower_volt) & (stim_df['AuxPhidgAO1'] < upper_volt)].index
    start_list = []
    end_list = []
    for id_num, idx in enumerate(indices):
        if np.isin(idx - 1, indices, invert=True) and np.isin(idx + 1, indices) and (id_num != len(indices) - 1):
            start_list.append(stim_df.iloc[[idx]]['timeVec'].values[0])
        elif np.isin(idx + 1, indices, invert=True) and np.isin(idx - 1, indices):
            end_list.append(stim_df.iloc[[idx]]['timeVec'].values[0])
    return (start_list, end_list)

def get_times_flight_stop_start(stimulus_df, flying_threshold=0.11, bout_time=5, frames_consecutive=5, seconds_pre=1):
    """
    return arrays start and stop indices 
    flying_threshold: threshold below indicates no flight
    frames_consecutive: how many consecutive frames to count as flight stop/flight start
    can include airpuffs (for now)
    no restriction on bout times (for now)
    """
    start_flight_times = []
    stop_flight_times = []
    all_times = stimulus_df['timeVec'].values
    for t_ind, time in enumerate(all_times):
        pre_t = time - seconds_pre
        post_t = time + bout_time
        front_check = pre_t < np.min(stimulus_df['timeVec'].values)
        back_check = post_t > np.max(stimulus_df['timeVec'].values)
        if front_check or back_check:
            continue
        else:
            curr_aux_val = stimulus_df[stimulus_df['timeVec'] == time]['aux_kinefly_filt'].values[0]
            curr_aux_idx = stimulus_df[stimulus_df['timeVec'] == time].index[0]
            prev_frames = int(curr_aux_idx - frames_consecutive)
            next_frames = int(curr_aux_idx + frames_consecutive)
            if curr_aux_val >= flying_threshold and (stimulus_df.iloc[prev_frames:curr_aux_idx]['aux_kinefly_filt'] < flying_threshold).all() and (stimulus_df.iloc[curr_aux_idx:next_frames]['aux_kinefly_filt'] >= flying_threshold).all():
                start_flight_times.append(time)
            elif curr_aux_val < flying_threshold and (stimulus_df.iloc[prev_frames:curr_aux_idx]['aux_kinefly_filt'] >= flying_threshold).all() and (stimulus_df.iloc[curr_aux_idx:next_frames]['aux_kinefly_filt'] < flying_threshold).all():
                stop_flight_times.append(time)
            else:
                pass
    return (start_flight_times, stop_flight_times)

def plot_avg_trace_per_visual_stimulus_single_experiment(datasets, patterns_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, plot_original_exp=True, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, wing_row=1, LRwing_row=2):
    """
    generate avg trace plots for each visual stimuli from either panel experiment or translation experiment
    combine left and right stimuli into side agnostic 
    single roi for dff
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 3
    columns_stim = 8
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    if plot_original_exp == False:
        exp_name = 'translation_experiment'
        stimulus_seconds = 3
        all_dFF_df, all_stim_df = load_combined(datasets['data_trans_df_dir'], line)
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
        frames_pre = int(imRateHz_mean * seconds_pre)
        frames_post = int(imRateHz_mean * seconds_post)
        frames_stim = int(imRateHz_mean * stimulus_seconds)
        for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
            dFF_all = []
            Lwing_all = []
            Rwing_all = []
            LpRwing_all = []
            for fly_id in all_dFF_df['fly_id'].unique():
                stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
                dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
                dFF_fly = []
                Lwing_fly = []
                Rwing_fly = []
                LpRwing_fly = []
                for trial in dFF_df_fly['trial_num'].unique():
                    stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                    dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                    flying_threshold = flying_threshold_default
                    if len(stim_df) > 3000:
                        start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                        if len(start_times) != 0 and len(end_times) != 0:
                            for ind, start in enumerate(start_times):
                                start_t = start - seconds_pre
                                end = end_times[ind]
                                end_t = end + seconds_post
                                pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                                pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                                stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                                stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                                post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                                post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                                pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                                stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                                post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                                pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                                stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                                post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                                pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                                stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                                post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                                pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                                stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                                post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                                flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                                air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                                if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                    t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                                    t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                                    dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                                    dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                    baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                    t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                                    Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                                    Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                    Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                    Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                                    Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                    Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                    LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                                    LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                    LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                    dFF_fly.append(dFF_int - baseline_dFF_int)
                                    Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                                    Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                                    LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
                if len(dFF_fly) != 0:
                    dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                    dFF_all.append(dFF_fly_mean)
                    Lwing_all.append(Lwing_fly_mean)
                    Rwing_all.append(Rwing_fly_mean)
                    LpRwing_all.append(LpRwing_fly_mean)
            if len(dFF_all) != 0:
                dFF_mean = np.nanmean(dFF_all, axis=0)
                print(f'{patterns_to_plot[stim_keys]} fly n= {len(dFF_all)}')
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                if plot_bs_CI == True:
                    CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                axs[2, stim_counter].set_xticklabels(axs[2, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[2, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
                if stim_counter == 0:
                    axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[dff_row, stim_counter].set_yticks([-0.5, 0, 1, 2, 3])
                    axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                    axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                    axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].spines['top'].set_visible(False)
                    axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[wing_row, stim_counter].spines['right'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                    axs[dff_row, stim_counter].spines['top'].set_visible(False)
                    axs[dff_row, stim_counter].spines['right'].set_visible(False)
                    axs[2, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                    axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                    axs[2, stim_counter].set_xticklabels(axs[2, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                    axs[0, stim_counter].set_xticks([])
                    axs[1, stim_counter].set_xticks([])
                    axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                    axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                    axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 3.0))
                else:
                    axs[dff_row, stim_counter].set_yticks([])
                    axs[wing_row, stim_counter].set_yticks([])
                    axs[LRwing_row, stim_counter].set_yticks([])
                    axs[0, stim_counter].set_xticks([])
                    axs[1, stim_counter].set_xticks([])
                    axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                    axs[2, stim_counter].set_xticklabels(axs[2, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].spines['top'].set_visible(False)
                    axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[wing_row, stim_counter].spines['right'].set_visible(False)
                    axs[wing_row, stim_counter].spines['left'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                    axs[dff_row, stim_counter].spines['top'].set_visible(False)
                    axs[dff_row, stim_counter].spines['right'].set_visible(False)
                    axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[wing_row, stim_counter].set_ylim([-20, 12])
                axs[LRwing_row, stim_counter].set_ylim([-25, 15])
                axs[dff_row, stim_counter].set_ylim([-0.5, 3])
                if stimulus_seconds > 2:
                    axs[wing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                    axs[LRwing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                    axs[dff_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                    axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                    axs[2, stim_counter].set_xticklabels(axs[2, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[2, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[2, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[wing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[2, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
                stim_counter += 1
    else:
        exp_name = 'original_experiment'
        all_dFF_df, all_stim_df = load_combined(datasets['data_general_df_dir'], line)
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
        frames_pre = int(imRateHz_mean * seconds_pre)
        frames_post = int(imRateHz_mean * seconds_post)
        Yaw_dFF_all = []
        Yaw_Outer_wing_all = []
        Yaw_Inner_wing_all = []
        Yaw_LpRwing_all = []
        Roll_dFF_all = []
        Roll_Outer_wing_all = []
        Roll_Inner_wing_all = []
        Roll_LpRwing_all = []
        Loom_dFF_all = []
        Loom_Outer_wing_all = []
        Loom_Inner_wing_all = []
        Loom_LpRwing_all = []
        Yaw_dFF_L_stim = []
        Yaw_Outer_wing_L_stim = []
        Yaw_Inner_wing_L_stim = []
        Yaw_LpRwing_L_stim = []
        Roll_dFF_L_stim = []
        Roll_Outer_wing_L_stim = []
        Roll_Inner_wing_L_stim = []
        Roll_LpRwing_L_stim = []
        Loom_dFF_L_stim = []
        Loom_Outer_wing_L_stim = []
        Loom_Inner_wing_L_stim = []
        Loom_LpRwing_L_stim = []
        Yaw_dFF_R_stim = []
        Yaw_Outer_wing_R_stim = []
        Yaw_Inner_wing_R_stim = []
        Yaw_LpRwing_R_stim = []
        Roll_dFF_R_stim = []
        Roll_Outer_wing_R_stim = []
        Roll_Inner_wing_R_stim = []
        Roll_LpRwing_R_stim = []
        Loom_dFF_R_stim = []
        Loom_Outer_wing_R_stim = []
        Loom_Inner_wing_R_stim = []
        Loom_LpRwing_R_stim = []
        for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
            dFF_all = []
            Lwing_all = []
            Rwing_all = []
            LpRwing_all = []
            for fly_id in all_dFF_df['fly_id'].unique():
                stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
                dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
                dFF_fly = []
                Lwing_fly = []
                Rwing_fly = []
                LpRwing_fly = []
                for trial in dFF_df_fly['trial_num'].unique():
                    stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                    dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                    flying_threshold = flying_threshold_default
                    if stim_keys >= 7.7:
                        start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=False)
                    else:
                        start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.25, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        if len(dFF_all) == 0:
                            stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                            frames_stim = int(imRateHz_mean * stimulus_seconds)
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                dFF_fly.append(dFF_int - baseline_dFF_int)
                                Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                                Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                                LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
                if len(dFF_fly) != 0:
                    dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                    dFF_all.append(dFF_fly_mean)
                    Lwing_all.append(Lwing_fly_mean)
                    Rwing_all.append(Rwing_fly_mean)
                    LpRwing_all.append(LpRwing_fly_mean)
            if len(dFF_all) != 0:
                if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw' and patterns_to_plot[stim_keys].split(' ')[0] != 'Roll' and (patterns_to_plot[stim_keys].split(' ')[0] != 'Loom'):
                    dFF_mean = np.nanmean(dFF_all, axis=0)
                    Lwing_mean = np.nanmean(Lwing_all, axis=0)
                    Rwing_mean = np.nanmean(Rwing_all, axis=0)
                    LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                    if plot_bs_CI == True:
                        CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                        CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                        CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                        CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                        axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                    axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                    axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                    axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
                    axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
                    axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                    axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
                else:
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                        if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                            Yaw_dFF_L_stim.append(dFF_all)
                            Yaw_Outer_wing_L_stim.append(Rwing_all)
                            Yaw_Inner_wing_L_stim.append(Lwing_all)
                            Yaw_LpRwing_L_stim.append(LpRwing_all)
                        else:
                            Yaw_dFF_R_stim.append(dFF_all)
                            Yaw_Outer_wing_R_stim.append(Lwing_all)
                            Yaw_Inner_wing_R_stim.append(Rwing_all)
                            Yaw_LpRwing_R_stim.append(LpRwing_all)
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                        if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                            Roll_dFF_L_stim.append(dFF_all)
                            Roll_Outer_wing_L_stim.append(Rwing_all)
                            Roll_Inner_wing_L_stim.append(Lwing_all)
                            Roll_LpRwing_L_stim.append(LpRwing_all)
                        else:
                            Roll_dFF_R_stim.append(dFF_all)
                            Roll_Outer_wing_R_stim.append(Lwing_all)
                            Roll_Inner_wing_R_stim.append(Rwing_all)
                            Roll_LpRwing_R_stim.append(LpRwing_all)
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                        if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                            Loom_dFF_L_stim.append(dFF_all)
                            Loom_Outer_wing_L_stim.append(Rwing_all)
                            Loom_Inner_wing_L_stim.append(Lwing_all)
                            Loom_LpRwing_L_stim.append(LpRwing_all)
                        else:
                            Loom_dFF_R_stim.append(dFF_all)
                            Loom_Outer_wing_R_stim.append(Lwing_all)
                            Loom_Inner_wing_R_stim.append(Rwing_all)
                            Loom_LpRwing_R_stim.append(LpRwing_all)
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                        if len(Yaw_dFF_L_stim) == 0 or len(Yaw_dFF_R_stim) == 0:
                            continue
                        else:
                            Yaw_dFF_all = [np.vstack(Yaw_dFF_L_stim), np.vstack(Yaw_dFF_R_stim)]
                            Yaw_dFF_all = np.vstack(Yaw_dFF_all)
                            Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                            Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                            Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                            Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                            Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                            Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                            dFF_mean = np.nanmean(Yaw_dFF_all, axis=0)
                            Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                            Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                            LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                            if plot_bs_CI == True:
                                CI_dFF = create_CI_timetrace(np.vstack(Yaw_dFF_all), iterations=1000, CI=95.0)
                                CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                                CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                                CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                            axs[0, stim_counter].set_title('Yaw\n ', fontsize=axis_label_fontsize)
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                        if len(Roll_dFF_L_stim) == 0 or len(Roll_dFF_R_stim) == 0:
                            continue
                        else:
                            Roll_dFF_all = [np.vstack(Roll_dFF_L_stim), np.vstack(Roll_dFF_R_stim)]
                            Roll_dFF_all = np.vstack(Roll_dFF_all)
                            Roll_Outer_wing_all = [np.vstack(Roll_Outer_wing_L_stim), np.vstack(Roll_Outer_wing_R_stim)]
                            Roll_Outer_wing_all = np.vstack(Roll_Outer_wing_all)
                            Roll_Inner_wing_all = [np.vstack(Roll_Inner_wing_L_stim), np.vstack(Roll_Inner_wing_R_stim)]
                            Roll_Inner_wing_all = np.vstack(Roll_Inner_wing_all)
                            Roll_LpRwing_all = [np.vstack(Roll_LpRwing_L_stim), np.vstack(Roll_LpRwing_R_stim)]
                            Roll_LpRwing_all = np.vstack(Roll_LpRwing_all)
                            dFF_mean = np.nanmean(Roll_dFF_all, axis=0)
                            Iwing_mean = np.nanmean(Roll_Inner_wing_all, axis=0)
                            Owing_mean = np.nanmean(Roll_Outer_wing_all, axis=0)
                            LpRwing_mean = np.nanmean(Roll_LpRwing_all, axis=0)
                            if plot_bs_CI == True:
                                CI_dFF = create_CI_timetrace(np.vstack(Roll_dFF_all), iterations=1000, CI=95.0)
                                CI_Iwing = create_CI_timetrace(np.vstack(Roll_Inner_wing_all), iterations=1000, CI=95.0)
                                CI_Owing = create_CI_timetrace(np.vstack(Roll_Outer_wing_all), iterations=1000, CI=95.0)
                                CI_lrwing = create_CI_timetrace(np.vstack(Roll_LpRwing_all), iterations=1000, CI=95.0)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                            axs[0, stim_counter].set_title('Roll\n ', fontsize=axis_label_fontsize)
                    if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                        if len(Loom_dFF_L_stim) == 0 or len(Loom_dFF_R_stim) == 0:
                            continue
                        else:
                            Loom_dFF_all = [np.vstack(Loom_dFF_L_stim), np.vstack(Loom_dFF_R_stim)]
                            Loom_dFF_all = np.vstack(Loom_dFF_all)
                            Loom_Outer_wing_all = [np.vstack(Loom_Outer_wing_L_stim), np.vstack(Loom_Outer_wing_R_stim)]
                            Loom_Outer_wing_all = np.vstack(Loom_Outer_wing_all)
                            Loom_Inner_wing_all = [np.vstack(Loom_Inner_wing_L_stim), np.vstack(Loom_Inner_wing_R_stim)]
                            Loom_Inner_wing_all = np.vstack(Loom_Inner_wing_all)
                            Loom_LpRwing_all = [np.vstack(Loom_LpRwing_L_stim), np.vstack(Loom_LpRwing_R_stim)]
                            Loom_LpRwing_all = np.vstack(Loom_LpRwing_all)
                            dFF_mean = np.nanmean(Loom_dFF_all, axis=0)
                            Iwing_mean = np.nanmean(Loom_Inner_wing_all, axis=0)
                            Owing_mean = np.nanmean(Loom_Outer_wing_all, axis=0)
                            LpRwing_mean = np.nanmean(Loom_LpRwing_all, axis=0)
                            if plot_bs_CI == True:
                                CI_dFF = create_CI_timetrace(np.vstack(Loom_dFF_all), iterations=1000, CI=95.0)
                                CI_Iwing = create_CI_timetrace(np.vstack(Loom_Inner_wing_all), iterations=1000, CI=95.0)
                                CI_Owing = create_CI_timetrace(np.vstack(Loom_Outer_wing_all), iterations=1000, CI=95.0)
                                CI_lrwing = create_CI_timetrace(np.vstack(Loom_LpRwing_all), iterations=1000, CI=95.0)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                                axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                            axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                            axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                axs[wing_row, stim_counter].set_ylim([-20, 12])
                axs[LRwing_row, stim_counter].set_ylim([-25, 15])
                axs[dff_row, stim_counter].set_ylim([-0.5, 3.0])
                axs[wing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[LRwing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[2, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
                if stim_counter == 0:
                    axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[dff_row, stim_counter].set_yticks([-0.5, 0, 1, 2, 3])
                    axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                    axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                    axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].spines['top'].set_visible(False)
                    axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[wing_row, stim_counter].spines['right'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                    axs[dff_row, stim_counter].spines['top'].set_visible(False)
                    axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[dff_row, stim_counter].spines['right'].set_visible(False)
                    axs[2, stim_counter].set_xlabel('time (s)', fontsize=axis_label_fontsize, ha='center')
                    axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                    axs[2, stim_counter].set_xticklabels(axs[2, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                    axs[0, stim_counter].set_xticks([])
                    axs[1, stim_counter].set_xticks([])
                    axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                    axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                    axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 3))
                else:
                    axs[dff_row, stim_counter].set_yticks([])
                    axs[wing_row, stim_counter].set_yticks([])
                    axs[LRwing_row, stim_counter].set_yticks([])
                    axs[0, stim_counter].set_xticks([])
                    axs[1, stim_counter].set_xticks([])
                    axs[2, stim_counter].set_xticks([0, stimulus_seconds])
                    axs[2, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                    axs[wing_row, stim_counter].spines['top'].set_visible(False)
                    axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[wing_row, stim_counter].spines['right'].set_visible(False)
                    axs[wing_row, stim_counter].spines['left'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                    axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                    axs[dff_row, stim_counter].spines['top'].set_visible(False)
                    axs[dff_row, stim_counter].spines['right'].set_visible(False)
                    axs[dff_row, stim_counter].spines['left'].set_visible(False)
                    axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[2, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[2, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[wing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
                axs[2, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
                stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'], 2.5)
        fig.subplots_adjust(left=0.076, bottom=0.117, right=0.988, top=0.883, wspace=0, hspace=0.114)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_updown_experiment_scale_up_down(datasets, patterns_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=3, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, LRwing_row=1, interp_to='median'):
    """
    generate avg trace plots for each visual stimuli from either up / down or up /down of various speeds 
    sep subplots for up vs static down
    combine left and right stimuli into side agnostic 
    *no L,R only L+R
    single roi for dff
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    interp to: median or min (if 2 different imaging rates)
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 2
    columns_stim = len(patterns_to_plot)
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            if stim_keys < 0:
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -3.0, 3.0, color=_settings['upward_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -40, 22.0, color=_settings['upward_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            else:
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -3.0, 3.0, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -40, 22.0, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
            axs[0, stim_counter].set_title(f'{patterns_to_plot[stim_keys]}\nn = {len(dFF_all)}', fontsize=axis_label_fontsize)
            axs[dff_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_keys < 0:
                axs[dff_row, stim_counter].set_yticks([-0.5, 0])
                axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 0))
                axs[dff_row, stim_counter].set_ylim([-0.5, 0.5])
            else:
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                axs[dff_row, stim_counter].set_ylim([-3.0, 3.0])
            if stimulus_seconds > 2:
                axs[LRwing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_yticks([-30, -20, -10, 0, 10, 20])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[0, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-30, 20))
            elif stim_keys == 6.0:
                axs[dff_row, stim_counter].spines['left'].set_visible(True)
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                axs[dff_row, stim_counter].set_ylim([-3.0, 3.0])
                axs[dff_row, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].set_xticks([])
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                axs[0, stim_counter].set_xticks([])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].set_ylim([-40, 22])
            axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.6, _settings['column_width_2'] * 0.25)
        fig.subplots_adjust(left=0.051, bottom=0.117, right=0.988, top=0.851, wspace=0.05, hspace=0.146)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_updown_experiment_LPT_combine_yaw_only_LpR_rescaled(datasets, patterns_to_plot, line='2075', flying_threshold_default=0.14, seconds_pre=1, seconds_post=3, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, LRwing_row=1, interp_to='median', quiescence=False, color_dff=None):
    """
    generate avg trace plots for each visual stimuli from either up / down 
    only include L+R row not individual L,R traces
    combine left and right stimuli into side agnostic 
    single roi for dff
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    interp to: median or min (if 2 different imaging rates)
    color_dff = if want to change the color of the dff trace, if None standard dff_color
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 2
    columns_stim = len(patterns_to_plot) - 1
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    Yaw_dFF_all = []
    Yaw_Outer_wing_all = []
    Yaw_Inner_wing_all = []
    Yaw_LpRwing_all = []
    Yaw_dFF_L_stim = []
    Yaw_Outer_wing_L_stim = []
    Yaw_Inner_wing_L_stim = []
    Yaw_LpRwing_L_stim = []
    Yaw_dFF_R_stim = []
    Yaw_Outer_wing_R_stim = []
    Yaw_Inner_wing_R_stim = []
    Yaw_LpRwing_R_stim = []
    if color_dff is not None:
        dFF_color = color_dff
    else:
        dFF_color = _settings['dFF_hexcode']
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    if len(dFF_all) == 0:
                        stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                        frames_stim = int(imRateHz_mean * stimulus_seconds)
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if quiescence == False:
                            flight_stop_bool = flight_stops_num == 0
                            air_puff_bool = air_puffs_num == 0
                        else:
                            flight_stop_bool = True
                            air_puff_bool = True
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and flight_stop_bool and air_puff_bool:
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw':
                dFF_mean = np.nanmean(dFF_all, axis=0)
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -40, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -40, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                if plot_bs_CI == True:
                    CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R WSA\n(deg)', linewidth=line_width)
                print(f'{patterns_to_plot[stim_keys]} N={len(dFF_all)}')
                axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
            else:
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Yaw_dFF_L_stim.append(dFF_all)
                        Yaw_Outer_wing_L_stim.append(Rwing_all)
                        Yaw_Inner_wing_L_stim.append(Lwing_all)
                        Yaw_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Yaw_dFF_R_stim.append(dFF_all)
                        Yaw_Outer_wing_R_stim.append(Lwing_all)
                        Yaw_Inner_wing_R_stim.append(Rwing_all)
                        Yaw_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if len(Yaw_dFF_L_stim) == 0 or len(Yaw_dFF_R_stim) == 0:
                        continue
                    else:
                        Yaw_dFF_all = [np.vstack(Yaw_dFF_L_stim), np.vstack(Yaw_dFF_R_stim)]
                        Yaw_dFF_all = np.vstack(Yaw_dFF_all)
                        Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                        Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                        Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                        Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                        Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                        Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                        dFF_mean = np.nanmean(Yaw_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -40, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -40, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Yaw_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
                        print(f'Yaw N={len(Yaw_dFF_all)}')
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R WSA\n(deg)', linewidth=line_width)
                        axs[0, stim_counter].set_title('Yaw', fontsize=axis_label_fontsize)
                        axs[dff_row, stim_counter].set_yticks([])
        axs[LRwing_row, stim_counter].set_yticks([])
        axs[1, stim_counter].set_xticks([0, stimulus_seconds])
        axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
        axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
        axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
        axs[dff_row, stim_counter].spines['top'].set_visible(False)
        axs[dff_row, stim_counter].spines['right'].set_visible(False)
        axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
        axs[1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
        if patterns_to_plot[stim_keys] == 'Pitch Up' or patterns_to_plot[stim_keys] == 'Upward':
            axs[dff_row, stim_counter].set_yticks([-0.5, 0, 0.5])
            axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 0.5))
            axs[dff_row, stim_counter].set_ylim([-0.6, 0.6])
        else:
            axs[dff_row, stim_counter].set_yticks([0, 5])
            axs[dff_row, stim_counter].spines.left.set_bounds((0, 5.0))
            axs[dff_row, stim_counter].set_ylim([-5.0, 5.5])
        if stim_counter == 0:
            axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
            axs[LRwing_row, stim_counter].set_ylabel('L+R WSA\n(deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
            axs[LRwing_row, stim_counter].set_yticks([-30, 0, 20])
            axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[0, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines.left.set_bounds((-30, 20))
        elif patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
            axs[dff_row, stim_counter].spines['left'].set_visible(True)
            axs[dff_row, stim_counter].set_yticks([0, 5])
            axs[dff_row, stim_counter].spines.left.set_bounds((0, 5.0))
            axs[dff_row, stim_counter].set_ylim([-5.0, 5.5])
            axs[dff_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
            axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
        else:
            axs[dff_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[0, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['left'].set_visible(False)
            axs[LRwing_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
        axs[LRwing_row, stim_counter].set_ylim([-40, 22])
        if stimulus_seconds > 2:
            axs[LRwing_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
            axs[dff_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
        axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
        stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.4, _settings['column_width_2'] * 0.15)
        fig.subplots_adjust(left=0.124, bottom=0.08, right=0.988, top=0.883, wspace=0, hspace=0.162)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    if quiescence:
        flight_str = 'quiescence'
    else:
        flight_str = 'flight'
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_updown_experiment_LPT_width_speed_LpR_only(datasets, patterns_to_plot, line='2075', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, LRwing_row=2, interp_to='median', exp_type='speed', color_dff=None):
    """
    generate avg trace plots for each visual stimuli from up /down of various speeds or varying width
    single roi for dff (LPT49)
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    interp to: median or min (if 2 different imaging rates)
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 2
    columns_stim = len(patterns_to_plot)
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    if color_dff is not None:
        dFF_color = color_dff
    else:
        dFF_color = _settings['dFF_hexcode']
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            if stim_keys < 0 and exp_type == 'speed':
                fill_color = _settings['upward_stim_hexcode']
            else:
                fill_color = _settings['visual_stim_hexcode']
            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=fill_color, alpha=alpha_patch, linewidth=0.0001)
            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=fill_color, alpha=alpha_patch, linewidth=0.0001)
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
            axs[0, stim_counter].set_title(f'{patterns_to_plot[stim_keys]}', fontsize=axis_label_fontsize)
            print(f'{patterns_to_plot[stim_keys]}\nn = {len(dFF_all)}')
            axs[dff_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stimulus_seconds > 2:
                axs[LRwing_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R WSA\ndeg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 2.5, 5])
                axs[LRwing_row, stim_counter].set_yticks([-30, -20, -10, 0, 10, 20])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[0, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-30, 20))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 5.0))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                axs[0, stim_counter].set_xticks([])
                axs[1, stim_counter].set_xticks([])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].set_ylim([-45, 22])
            axs[dff_row, stim_counter].set_ylim([-0.5, 6.0])
            axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if exp_type == 'speed':
        column_width = _settings['column_width_2'] * 0.38
    else:
        column_width = _settings['column_width_2'] * 0.16
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(column_width, _settings['column_width_2'] * 0.25)
        fig.subplots_adjust(left=0.11, bottom=0.117, right=0.988, top=0.883, wspace=0, hspace=0.114)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_updown_experiment_LPT_width_speed_LpR_only_rescaled(datasets, patterns_to_plot, line='2075', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, LRwing_row=2, interp_to='median', exp_type='speed', color_dff=None):
    """
    generate avg trace plots for each visual stimuli from up /down of various speeds or varying width
    single roi for dff (LPT49)
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    interp to: median or min (if 2 different imaging rates)
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 2
    columns_stim = len(patterns_to_plot)
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    if color_dff is not None:
        dFF_color = color_dff
    else:
        dFF_color = _settings['dFF_hexcode']
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            if stim_keys < 0 and exp_type == 'speed':
                fill_color = _settings['upward_stim_hexcode']
            else:
                fill_color = _settings['visual_stim_hexcode']
            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=fill_color, alpha=alpha_patch, linewidth=0.0001)
            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=fill_color, alpha=alpha_patch, linewidth=0.0001)
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
            axs[0, stim_counter].set_title(f'{patterns_to_plot[stim_keys]}', fontsize=axis_label_fontsize)
            print(f'{patterns_to_plot[stim_keys]}\nn = {len(dFF_all)}')
            axs[dff_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_keys < 0:
                axs[dff_row, stim_counter].set_yticks([-0.5, 0])
                axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 0))
                axs[dff_row, stim_counter].set_ylim([-0.6, 0.6])
            else:
                axs[dff_row, stim_counter].set_yticks([0, 2.5, 5])
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 5))
                axs[dff_row, stim_counter].set_ylim([-5.5, 5.5])
            if stimulus_seconds > 2:
                axs[LRwing_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R WSA\ndeg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_yticks([-30, -20, -10, 0, 10, 20])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[0, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-30, 20))
            elif patterns_to_plot[stim_keys] == 'Static Image':
                axs[dff_row, stim_counter].spines['left'].set_visible(True)
                axs[dff_row, stim_counter].set_yticks([0, 2.5, 5])
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 5.0))
                axs[dff_row, stim_counter].set_ylim([-5.5, 5.5])
                axs[dff_row, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                axs[0, stim_counter].set_xticks([])
                axs[1, stim_counter].set_xticks([])
                axs[1, stim_counter].set_xticks([0, stimulus_seconds])
                axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].set_xticks([])
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].set_ylim([-45, 22])
            axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if exp_type == 'speed':
        column_width = _settings['column_width_2'] * 0.38
    else:
        column_width = _settings['column_width_2'] * 0.16
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(column_width, _settings['column_width_2'] * 0.25)
        fig.subplots_adjust(left=0.11, bottom=0.117, right=0.988, top=0.883, wspace=0.145, hspace=0.114)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_updown_experiment_combine_yaw_avg_roi_LpR_only_rescaled(datasets, patterns_to_plot, line='2087', flying_threshold_default=0.14, seconds_pre=1, seconds_post=3, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, LRwing_row=1, interp_to='median', quiescence=False, color_dff=None):
    """
    generate avg trace plots for each visual stimuli from either up / down or up /down of various speeds 
    combine left and right stimuli into side agnostic (for yaw)
    avg 2 rois (L and R if applicable)
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    interp to: median or min (if 2 different imaging rates)
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 2
    columns_stim = len(patterns_to_plot) - 1
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
        imRate_str = ''
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
        imRate_str = ''
    elif isinstance(interp_to, float):
        imRateHz_mean = interp_to
        imRate_str = '_ds'
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
        imRate_str = ''
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    Yaw_dFF_all = []
    Yaw_Outer_wing_all = []
    Yaw_Inner_wing_all = []
    Yaw_LpRwing_all = []
    Yaw_dFF_L_stim = []
    Yaw_Outer_wing_L_stim = []
    Yaw_Inner_wing_L_stim = []
    Yaw_LpRwing_L_stim = []
    Yaw_dFF_R_stim = []
    Yaw_Outer_wing_R_stim = []
    Yaw_Inner_wing_R_stim = []
    Yaw_LpRwing_R_stim = []
    if color_dff is not None:
        dFF_color = color_dff
    else:
        dFF_color = _settings['dFF_hexcode']
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    if len(dFF_all) == 0:
                        stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                        frames_stim = int(imRateHz_mean * stimulus_seconds)
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF_left = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_left'].values
                        pre_dFF_right = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_right'].values
                        pre_dFF = np.nanmean(np.vstack([pre_dFF_left, pre_dFF_right]), axis=0)
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF_left = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF_left'].values
                        stim_dFF_right = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF_right'].values
                        stim_dFF = np.nanmean(np.vstack([stim_dFF_left, stim_dFF_right]), axis=0)
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF_left = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF_left'].values
                        post_dFF_right = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF_right'].values
                        post_dFF = np.nanmean(np.vstack([post_dFF_left, post_dFF_right]), axis=0)
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if quiescence == False:
                            flight_stop_bool = flight_stops_num == 0
                            air_puff_bool = air_puffs_num == 0
                        else:
                            flight_stop_bool = True
                            air_puff_bool = True
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and flight_stop_bool and air_puff_bool:
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw':
                dFF_mean = np.nanmean(dFF_all, axis=0)
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                if plot_bs_CI == True:
                    CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R WSA\n(deg)', linewidth=line_width)
                axs[0, stim_counter].set_title(f'{patterns_to_plot[stim_keys]}', fontsize=axis_label_fontsize)
                print(f'{patterns_to_plot[stim_keys]} N={len(dFF_all)}')
            else:
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Yaw_dFF_L_stim.append(dFF_all)
                        Yaw_Outer_wing_L_stim.append(Rwing_all)
                        Yaw_Inner_wing_L_stim.append(Lwing_all)
                        Yaw_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Yaw_dFF_R_stim.append(dFF_all)
                        Yaw_Outer_wing_R_stim.append(Lwing_all)
                        Yaw_Inner_wing_R_stim.append(Rwing_all)
                        Yaw_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if len(Yaw_dFF_L_stim) == 0 or len(Yaw_dFF_R_stim) == 0:
                        continue
                    else:
                        Yaw_dFF_all = [np.vstack(Yaw_dFF_L_stim), np.vstack(Yaw_dFF_R_stim)]
                        Yaw_dFF_all = np.vstack(Yaw_dFF_all)
                        Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                        Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                        Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                        Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                        Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                        Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                        dFF_mean = np.nanmean(Yaw_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -45, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Yaw_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=dFF_color, alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=dFF_color, linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R WSA\n(deg)', linewidth=line_width)
                        axs[0, stim_counter].set_title(f'Yaw', fontsize=axis_label_fontsize)
                        print(f'Yaw left N={np.shape(Yaw_dFF_L_stim)[1]}\nYaw right N={np.shape(Yaw_dFF_R_stim)[1]}')
                        axs[dff_row, stim_counter].set_yticks([])
        axs[1, stim_counter].set_xticks([0, stimulus_seconds])
        axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
        axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
        axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
        axs[dff_row, stim_counter].spines['top'].set_visible(False)
        axs[dff_row, stim_counter].spines['right'].set_visible(False)
        axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
        axs[1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
        if patterns_to_plot[stim_keys] == 'Pitch Up' or patterns_to_plot[stim_keys] == 'Upward':
            axs[dff_row, stim_counter].set_yticks([-0.5, 0, 0.5])
            axs[dff_row, stim_counter].spines.left.set_bounds((-0.5, 0.5))
            axs[dff_row, stim_counter].set_ylim([-0.6, 0.6])
        else:
            axs[dff_row, stim_counter].set_yticks([0, 2])
            axs[dff_row, stim_counter].spines.left.set_bounds((0, 2.0))
            axs[dff_row, stim_counter].set_ylim([-2.5, 2.5])
        if stimulus_seconds > 2:
            axs[LRwing_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
            axs[dff_row, stim_counter].set_xlim([-1, stimulus_seconds + seconds_post])
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
        if stim_counter == 0:
            axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
            axs[LRwing_row, stim_counter].set_ylabel('L+R WSA\n(deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
            axs[LRwing_row, stim_counter].set_yticks([-30, 0, 20])
            axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[0, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines.left.set_bounds((-30, 20))
        elif patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
            axs[dff_row, stim_counter].spines['left'].set_visible(True)
            axs[dff_row, stim_counter].set_yticks([0, 2])
            axs[dff_row, stim_counter].spines.left.set_bounds((0, 2.0))
            axs[dff_row, stim_counter].set_ylim([-2.5, 2.5])
            axs[dff_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
            axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].set_xticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
        else:
            axs[dff_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[0, stim_counter].set_xticks([])
            axs[1, stim_counter].set_xticks([])
            axs[1, stim_counter].set_xticks([0, stimulus_seconds])
            axs[1, stim_counter].set_xticklabels(axs[1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['left'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[LRwing_row, stim_counter].set_xticks([])
        axs[LRwing_row, stim_counter].set_ylim([-45, 22])
        axs[0, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[0, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[1, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[1, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
        stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.4, _settings['column_width_2'] * 0.15)
        fig.subplots_adjust(left=0.124, bottom=0.08, right=0.988, top=0.883, wspace=0, hspace=0.162)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    if quiescence:
        flight_str = 'quiescence'
    else:
        flight_str = 'flight'
    return dict(figure=fig, results={})

def plot_tuning_curve_updown_experiments(datasets, patterns_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, line_width=2, dff_col=0, LRwing_col=1, stat_method='max', response_curve_to_compare=-1, interp_to='median', plot_fly_means=False, xloc_key={-5.5: -9, -4.0: -6.0, -3.0: -3.0, -2.0: -2.0, -1.0: -1.0, 6.0: 0, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 6.0, 5.5: 9.0}, xlabel_key={-5.5: '+112.5', -4.0: '+75.0', -3.0: '+37.5', -2.0: '+25.0', -1.0: '+12.5', 6.0: '0.0', 1.0: '-12.5', 2.0: '-25.0', 3.0: '-37.5', 4.0: '-75.0', 5.5: '-112.5'}, point_size=6, jitter_means=True):
    """
    generate tuning curve for up / down stimuli of various speeds for dff and L+R
    single roi for dff
    mean of fly means 
    bs CI  
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    still post period (for determining flight stops and airpuffs shortly after stimulus period)
    stat method: 'mean', 'median', or 'max' response during stimulus period
    response_curve_to_compare: which part of stimulus to take peak response from (-x means last x seconds)
    interp to: median or min (if 2 different imaging rates)
    plot_fly_means: include individual fly means in plot
    """
    _settings = config.load('two_photon')['settings']
    rows_stim = 1
    columns_stim = 2
    fig, axs = plt.subplots(rows_stim, columns_stim, squeeze=False)
    stim_counter = 0
    row = 0
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if interp_to == 'median':
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    elif interp_to == 'min':
        imRateHz_mean = np.min(all_dFF_df['imRateHz'].values)
    else:
        imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    frames_for_avg = int(imRateHz_mean * response_curve_to_compare)
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and flight_stops_num == 0 and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end - start, frames_pre + frames_stim)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            stim_dFF_int = dFF_int[frames_pre:frames_pre + frames_stim]
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t)) - start
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            LpRstim_wing_int = LpRwing_int[frames_pre:frames_pre + frames_stim]
                            if stat_method == 'max':
                                if stim_keys < 0:
                                    dFF_fly.append(np.nanmin((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmin((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                else:
                                    dFF_fly.append(np.nanmax((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmax((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                            elif stat_method == 'mean':
                                dFF_fly.append(np.nanmean((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                LpRwing_fly.append(np.nanmean((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                            elif stat_method == 'median':
                                dFF_fly.append(np.nanmean((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                LpRwing_fly.append(np.nanmean((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                            else:
                                raise ValueError('stat method specified not valid')
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
                if plot_fly_means == True:
                    if jitter_means == True:
                        jitter = np.random.uniform(-0.1, 0.1, 1)
                        data_points = [dFF_fly_mean]
                        axs[row, dff_col].scatter(np.full_like(data_points, xloc_key[stim_keys]) + jitter, data_points, color='black', alpha=1.0, s=point_size, zorder=3, edgecolor='none', marker='o')
                        axs[row, dff_col].scatter(xloc_key[stim_keys], dFF_fly_mean, color='black', alpha=0.0, s=3)
                        data_points_LpR = [LpRwing_fly_mean]
                        axs[row, LRwing_col].scatter(np.full_like(data_points_LpR, xloc_key[stim_keys]) + jitter, data_points_LpR, color='black', alpha=1.0, s=point_size, zorder=3, edgecolor='none', marker='o')
                        axs[row, LRwing_col].scatter(xloc_key[stim_keys], LpRwing_fly_mean, color='black', alpha=0.0, s=3)
                    else:
                        axs[row, LRwing_col].scatter(xloc_key[stim_keys], LpRwing_fly_mean, alpha=1.0, color='black', s=point_size, marker='o', edgecolors='none')
                        axs[row, dff_col].scatter(xloc_key[stim_keys], dFF_fly_mean, alpha=1.0, color='black', s=point_size, marker='o', edgecolors='none')
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            if plot_bs_CI == True:
                CI_dFF = create_CI(dFF_all, CI=95.0, iterations=1000)
                CI_lrwing = create_CI(LpRwing_all, CI=95.0, iterations=1000)
                ci_height = CI_dFF[1] - CI_dFF[0]
                rect = Rectangle((xloc_key[stim_keys] - 0.25, CI_dFF[0]), width=0.5, height=ci_height, facecolor=_settings['bs_CI_hexcode'], alpha=0.5, edgecolor='none', zorder=1)
                axs[row, dff_col].add_patch(rect)
                axs[row, dff_col].hlines(dFF_mean, xloc_key[stim_keys] - 0.25, xloc_key[stim_keys] + 0.25, color='blue', lw=line_width, zorder=2)
                ci_height = CI_lrwing[1] - CI_lrwing[0]
                rect = Rectangle((xloc_key[stim_keys] - 0.25, CI_lrwing[0]), width=0.5, height=ci_height, facecolor=_settings['bs_CI_hexcode'], alpha=0.5, edgecolor='none', zorder=1)
                axs[row, LRwing_col].add_patch(rect)
                axs[row, LRwing_col].hlines(LpRwing_mean, xloc_key[stim_keys] - 0.25, xloc_key[stim_keys] + 0.25, color='blue', lw=line_width, zorder=2)
            else:
                axs[row, LRwing_col].scatter(xloc_key[stim_keys], LpRwing_mean, color='black', s=_settings['point_size_min'])
                axs[row, dff_col].scatter(xloc_key[stim_keys], dFF_mean, color='black', s=_settings['point_size_min'])
            print(f'{patterns_to_plot[stim_keys]} n = {len(dFF_all)}')
    tick_positions = [xloc_key[key] for key in patterns_to_plot]
    tick_labels = [f'{xlabel_key[key]}' for key in patterns_to_plot]
    axs[row, dff_col].set_xticks(tick_positions)
    axs[row, dff_col].set_xticklabels(tick_labels)
    axs[row, LRwing_col].set_xticks(tick_positions)
    axs[row, LRwing_col].set_xticklabels(tick_labels)
    axs[row, LRwing_col].spines.bottom.set_bounds((-9.0, 9.0))
    axs[row, LRwing_col].spines.left.set_bounds((-30, 20.0))
    axs[row, LRwing_col].set_xlim([-10, 9.5])
    axs[row, dff_col].spines.bottom.set_bounds((-9.0, 9.0))
    axs[row, dff_col].set_xlim([-10, 9.5])
    axs[row, dff_col].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center')
    axs[row, dff_col].spines['right'].set_visible(False)
    axs[row, dff_col].spines['top'].set_visible(False)
    axs[row, dff_col].set_xlabel('Speed Translational Motion', fontsize=axis_label_fontsize, ha='center')
    axs[row, LRwing_col].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, ha='center', labelpad=-5)
    axs[row, LRwing_col].spines['right'].set_visible(False)
    axs[row, LRwing_col].spines['top'].set_visible(False)
    axs[row, LRwing_col].set_ylim([-40, 25.0])
    axs[row, LRwing_col].set_yticks([-30, 0, 20])
    if stat_method == 'max':
        axs[row, dff_col].set_ylim([-1.0, 10.0])
        axs[row, dff_col].set_yticks([-0.5, 2, 4, 6, 8, 10])
        axs[row, dff_col].spines.left.set_bounds((-0.5, 10.0))
    else:
        axs[row, dff_col].set_ylim([-1.0, 4.0])
        axs[row, dff_col].set_yticks([-0.5, 0, 1, 2, 3, 4])
        axs[row, dff_col].spines.left.set_bounds((-0.5, 4.0))
    for column in range(columns_stim):
        axs[row, column].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[row, column].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[row, column].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[row, column].spines.bottom.set_linewidth(_settings['line_width_spines'])
        axs[row, column].set_xticklabels(axs[row, column].get_xticklabels(), fontsize=axis_ticklabel_fontsize, rotation=90)
        axs[row, column].set_yticklabels(axs[row, column].get_yticklabels(), fontsize=axis_ticklabel_fontsize)
    if jitter_means:
        jitter_str = '_jitter'
    else:
        jitter_str = ''
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.44, _settings['column_width_2'] * 0.26)
        fig.subplots_adjust(left=0.107, bottom=0.244, right=0.977, top=0.962, wspace=0.31, hspace=0.114)
    else:
        fig.set_size_inches(_settings['column_width_3'], 2.5)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_both_experiments(datasets, patterns_to_plot, patterns_to_plot_original, patterns_to_plot_trans, line='1070', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=1, wing_row=2, LRwing_row=3, include_loom_trace=True, loom_trace_row=0):
    """
    generate avg trace plots for each visual stimuli from either original experiment or trans experiment
    combine left and right stimuli into side agnostic 
    single roi for dff
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    """
    _settings = config.load('two_photon')['settings']
    if include_loom_trace == True:
        rows_stim = 4
        row_height = 3
    else:
        rows_stim = 3
        row_height = _settings['column_width_2'] * (1 / 4)
    columns_stim = 9
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    exp_name = 'combined_exp'
    all_dFF_df, all_stim_df = load_combined(datasets['data_general_df_dir'], line)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    Yaw_dFF_all = []
    Yaw_Outer_wing_all = []
    Yaw_Inner_wing_all = []
    Yaw_LpRwing_all = []
    Roll_dFF_all = []
    Roll_Outer_wing_all = []
    Roll_Inner_wing_all = []
    Roll_LpRwing_all = []
    Loom_dFF_all = []
    Loom_Outer_wing_all = []
    Loom_Inner_wing_all = []
    Loom_LpRwing_all = []
    Yaw_dFF_L_stim = []
    Yaw_Outer_wing_L_stim = []
    Yaw_Inner_wing_L_stim = []
    Yaw_LpRwing_L_stim = []
    Roll_dFF_L_stim = []
    Roll_Outer_wing_L_stim = []
    Roll_Inner_wing_L_stim = []
    Roll_LpRwing_L_stim = []
    Loom_dFF_L_stim = []
    Loom_Outer_wing_L_stim = []
    Loom_Inner_wing_L_stim = []
    Loom_LpRwing_L_stim = []
    Yaw_dFF_R_stim = []
    Yaw_Outer_wing_R_stim = []
    Yaw_Inner_wing_R_stim = []
    Yaw_LpRwing_R_stim = []
    Roll_dFF_R_stim = []
    Roll_Outer_wing_R_stim = []
    Roll_Inner_wing_R_stim = []
    Roll_LpRwing_R_stim = []
    Loom_dFF_R_stim = []
    Loom_Outer_wing_R_stim = []
    Loom_Inner_wing_R_stim = []
    Loom_LpRwing_R_stim = []
    for stim_idx, stim_keys in enumerate(patterns_to_plot_original.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if stim_keys >= 7.7:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=False)
                else:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.25, half_volt=True)
                if len(start_times) != 0 and len(end_times) != 0:
                    if len(dFF_all) == 0:
                        stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                        frames_stim = int(imRateHz_mean * stimulus_seconds)
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw' and patterns_to_plot[stim_keys].split(' ')[0] != 'Roll' and (patterns_to_plot[stim_keys].split(' ')[0] != 'Loom'):
                dFF_mean = np.nanmean(dFF_all, axis=0)
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                print(patterns_to_plot[stim_keys] + ': ' + str(len(LpRwing_all)))
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                if plot_bs_CI == True:
                    CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                if include_loom_trace == True and patterns_to_plot[stim_keys] == 'Front \nloom':
                    stimulus_seconds = _settings['loom_duration']
                axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
                if include_loom_trace == True:
                    if patterns_to_plot[stim_keys] == 'Front \nloom':
                        loom_radians = np.loadtxt(loom_path(), delimiter=',')
                        loom_deg = loom_radians * (180 / np.pi)
                        loom_deg[loom_deg < 0] = 0
                        loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                        loom_with_post_pre = np.concatenate((np.full(int(seconds_pre * _settings['loom_hz']), np.nan), loom_with_post))
                        t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                        axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                        axs[loom_trace_row, stim_counter].set_ylim([0, 90])
                        axs[loom_trace_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                        axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                        axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                        axs[loom_trace_row, stim_counter].set_yticks([])
                    else:
                        axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                        axs[loom_trace_row, stim_counter].set_yticks([])
            else:
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Yaw_dFF_L_stim.append(dFF_all)
                        Yaw_Outer_wing_L_stim.append(Rwing_all)
                        Yaw_Inner_wing_L_stim.append(Lwing_all)
                        Yaw_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Yaw_dFF_R_stim.append(dFF_all)
                        Yaw_Outer_wing_R_stim.append(Lwing_all)
                        Yaw_Inner_wing_R_stim.append(Rwing_all)
                        Yaw_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Roll_dFF_L_stim.append(dFF_all)
                        Roll_Outer_wing_L_stim.append(Rwing_all)
                        Roll_Inner_wing_L_stim.append(Lwing_all)
                        Roll_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Roll_dFF_R_stim.append(dFF_all)
                        Roll_Outer_wing_R_stim.append(Lwing_all)
                        Roll_Inner_wing_R_stim.append(Rwing_all)
                        Roll_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Loom_dFF_L_stim.append(dFF_all)
                        Loom_Outer_wing_L_stim.append(Rwing_all)
                        Loom_Inner_wing_L_stim.append(Lwing_all)
                        Loom_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Loom_dFF_R_stim.append(dFF_all)
                        Loom_Outer_wing_R_stim.append(Lwing_all)
                        Loom_Inner_wing_R_stim.append(Rwing_all)
                        Loom_LpRwing_R_stim.append(LpRwing_all)
                print(patterns_to_plot[stim_keys] + ': ' + str(len(LpRwing_all)))
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if len(Yaw_dFF_L_stim) == 0 or len(Yaw_dFF_R_stim) == 0:
                        continue
                    else:
                        Yaw_dFF_all = [np.vstack(Yaw_dFF_L_stim), np.vstack(Yaw_dFF_R_stim)]
                        Yaw_dFF_all = np.vstack(Yaw_dFF_all)
                        Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                        Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                        Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                        Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                        Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                        Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                        dFF_mean = np.nanmean(Yaw_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Yaw_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Yaw_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Yaw\n ', fontsize=axis_label_fontsize)
                        axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                        axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                        axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                        axs[dff_row, stim_counter].spines['left'].set_visible(False)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if len(Roll_dFF_L_stim) == 0 or len(Roll_dFF_R_stim) == 0:
                        continue
                    else:
                        Roll_dFF_all = [np.vstack(Roll_dFF_L_stim), np.vstack(Roll_dFF_R_stim)]
                        Roll_dFF_all = np.vstack(Roll_dFF_all)
                        Roll_Outer_wing_all = [np.vstack(Roll_Outer_wing_L_stim), np.vstack(Roll_Outer_wing_R_stim)]
                        Roll_Outer_wing_all = np.vstack(Roll_Outer_wing_all)
                        Roll_Inner_wing_all = [np.vstack(Roll_Inner_wing_L_stim), np.vstack(Roll_Inner_wing_R_stim)]
                        Roll_Inner_wing_all = np.vstack(Roll_Inner_wing_all)
                        Roll_LpRwing_all = [np.vstack(Roll_LpRwing_L_stim), np.vstack(Roll_LpRwing_R_stim)]
                        Roll_LpRwing_all = np.vstack(Roll_LpRwing_all)
                        dFF_mean = np.nanmean(Roll_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Roll_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Roll_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Roll_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Roll_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Roll_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Roll_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Roll_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Roll_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Roll\n ', fontsize=axis_label_fontsize)
                        axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                        axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                        axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                        axs[dff_row, stim_counter].spines['left'].set_visible(False)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if len(Loom_dFF_L_stim) == 0 or len(Loom_dFF_R_stim) == 0:
                        continue
                    else:
                        if include_loom_trace == True:
                            stimulus_seconds = _settings['loom_duration']
                        Loom_dFF_all = [np.vstack(Loom_dFF_L_stim), np.vstack(Loom_dFF_R_stim)]
                        Loom_dFF_all = np.vstack(Loom_dFF_all)
                        Loom_Outer_wing_all = [np.vstack(Loom_Outer_wing_L_stim), np.vstack(Loom_Outer_wing_R_stim)]
                        Loom_Outer_wing_all = np.vstack(Loom_Outer_wing_all)
                        Loom_Inner_wing_all = [np.vstack(Loom_Inner_wing_L_stim), np.vstack(Loom_Inner_wing_R_stim)]
                        Loom_Inner_wing_all = np.vstack(Loom_Inner_wing_all)
                        Loom_LpRwing_all = [np.vstack(Loom_LpRwing_L_stim), np.vstack(Loom_LpRwing_R_stim)]
                        Loom_LpRwing_all = np.vstack(Loom_LpRwing_all)
                        dFF_mean = np.nanmean(Loom_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Loom_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Loom_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Loom_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Loom_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Loom_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Loom_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Loom_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Loom_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                        if include_loom_trace == True:
                            loom_radians = np.loadtxt(loom_path(), delimiter=',')
                            loom_deg = loom_radians * (180 / np.pi)
                            loom_deg[loom_deg < 0] = 0
                            loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                            loom_with_post_pre = np.concatenate((np.full(int(seconds_pre * _settings['loom_hz']), np.nan), loom_with_post))
                            t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                            axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                            axs[loom_trace_row, stim_counter].set_ylim([0, 90])
                            axs[loom_trace_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
                            axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                            axs[loom_trace_row, stim_counter].set_ylabel('90 deg', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, va='bottom')
                            axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                            axs[loom_trace_row, stim_counter].set_yticks([])
                        axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                else:
                    axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                    axs[loom_trace_row, stim_counter].set_yticks([])
            axs[wing_row, stim_counter].set_ylim([-20, 12])
            axs[LRwing_row, stim_counter].set_ylim([-25, 15])
            axs[dff_row, stim_counter].set_ylim([-0.75, 3])
            axs[wing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[LRwing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[dff_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[rows_stim - 1, stim_counter].set_xlabel('time (s)', fontsize=axis_label_fontsize, va='center')
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[wing_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            for i in range(rows_stim):
                axs[i, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[i, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[wing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_trans_df_dir'], line)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    for stim_idx, stim_keys in enumerate(patterns_to_plot_trans.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                dFF_fly.append(dFF_int - baseline_dFF_int)
                                Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                                Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                                LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(patterns_to_plot_trans[stim_keys] + ': ' + str(len(LpRwing_all)))
            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
            axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
            axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
            axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
            axs[dff_row, stim_counter].set_yticks([])
            axs[wing_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds + seconds_post)])
            axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
            axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[wing_row, stim_counter].spines['top'].set_visible(False)
            axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[wing_row, stim_counter].spines['right'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[rows_stim - 1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[rows_stim - 1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[wing_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
            axs[wing_row, stim_counter].set_ylim([-20, 12])
            axs[LRwing_row, stim_counter].set_ylim([-25, 15])
            axs[dff_row, stim_counter].set_ylim([-0.75, 3])
            if stimulus_seconds > 2:
                axs[wing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[LRwing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            for i in range(rows_stim):
                axs[i, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[i, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            if include_loom_trace == True:
                axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                axs[loom_trace_row, stim_counter].set_yticks([])
            if include_loom_trace == False:
                axs[0, 0].spines['left'].set_visible(True)
            for i in range(rows_stim - 1):
                axs[i, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 2 / 3, row_height)
        fig.subplots_adjust(left=0.09, bottom=0.11, right=0.974, top=0.9, wspace=0.102, hspace=0.23)
    else:
        fig.set_size_inches(_settings['column_width_3'], row_height)
        fig.subplots_adjust(wspace=0.02, hspace=0.22)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_both_experiments_quiescence(datasets, patterns_to_plot, patterns_to_plot_original, patterns_to_plot_trans, line='1070', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=1, wing_row=2, LRwing_row=3, include_loom_trace=True, loom_trace_row=0):
    """
    generate avg trace plots for each visual stimuli from either original experiment or trans experiment
    combine left and right stimuli into side agnostic 
    single roi for dff
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    """
    _settings = config.load('two_photon')['settings']
    if include_loom_trace == True:
        rows_stim = 4
        row_height = 3
    else:
        rows_stim = 3
        row_height = _settings['column_width_2'] * (1 / 4)
    columns_stim = 9
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    exp_name = 'combined_exp'
    all_dFF_df, all_stim_df = load_combined(datasets['data_general_df_dir'], line)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    Yaw_dFF_all = []
    Yaw_Outer_wing_all = []
    Yaw_Inner_wing_all = []
    Yaw_LpRwing_all = []
    Roll_dFF_all = []
    Roll_Outer_wing_all = []
    Roll_Inner_wing_all = []
    Roll_LpRwing_all = []
    Loom_dFF_all = []
    Loom_Outer_wing_all = []
    Loom_Inner_wing_all = []
    Loom_LpRwing_all = []
    Yaw_dFF_L_stim = []
    Yaw_Outer_wing_L_stim = []
    Yaw_Inner_wing_L_stim = []
    Yaw_LpRwing_L_stim = []
    Roll_dFF_L_stim = []
    Roll_Outer_wing_L_stim = []
    Roll_Inner_wing_L_stim = []
    Roll_LpRwing_L_stim = []
    Loom_dFF_L_stim = []
    Loom_Outer_wing_L_stim = []
    Loom_Inner_wing_L_stim = []
    Loom_LpRwing_L_stim = []
    Yaw_dFF_R_stim = []
    Yaw_Outer_wing_R_stim = []
    Yaw_Inner_wing_R_stim = []
    Yaw_LpRwing_R_stim = []
    Roll_dFF_R_stim = []
    Roll_Outer_wing_R_stim = []
    Roll_Inner_wing_R_stim = []
    Roll_LpRwing_R_stim = []
    Loom_dFF_R_stim = []
    Loom_Outer_wing_R_stim = []
    Loom_Inner_wing_R_stim = []
    Loom_LpRwing_R_stim = []
    for stim_idx, stim_keys in enumerate(patterns_to_plot_original.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if stim_keys >= 7.7:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=False)
                else:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.25, half_volt=True)
                if len(start_times) != 0 and len(end_times) != 0:
                    if len(dFF_all) == 0:
                        stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                        frames_stim = int(imRateHz_mean * stimulus_seconds)
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_bouts_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values >= flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_bouts_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw' and patterns_to_plot[stim_keys].split(' ')[0] != 'Roll' and (patterns_to_plot[stim_keys].split(' ')[0] != 'Loom'):
                dFF_mean = np.nanmean(dFF_all, axis=0)
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                print(patterns_to_plot[stim_keys] + ': ' + str(len(LpRwing_all)))
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                if plot_bs_CI == True:
                    CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                if include_loom_trace == True and patterns_to_plot[stim_keys] == 'Front \nloom':
                    stimulus_seconds = _settings['loom_duration']
                axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
                if include_loom_trace == True:
                    if patterns_to_plot[stim_keys] == 'Front \nloom':
                        loom_radians = np.loadtxt(loom_path(), delimiter=',')
                        loom_deg = loom_radians * (180 / np.pi)
                        loom_deg[loom_deg < 0] = 0
                        loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                        loom_with_post_pre = np.concatenate((np.full(int(seconds_pre * _settings['loom_hz']), np.nan), loom_with_post))
                        t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                        axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                        axs[loom_trace_row, stim_counter].set_ylim([0, 90])
                        axs[loom_trace_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                        axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                        axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                        axs[loom_trace_row, stim_counter].set_yticks([])
                    else:
                        axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                        axs[loom_trace_row, stim_counter].set_yticks([])
            else:
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Yaw_dFF_L_stim.append(dFF_all)
                        Yaw_Outer_wing_L_stim.append(Rwing_all)
                        Yaw_Inner_wing_L_stim.append(Lwing_all)
                        Yaw_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Yaw_dFF_R_stim.append(dFF_all)
                        Yaw_Outer_wing_R_stim.append(Lwing_all)
                        Yaw_Inner_wing_R_stim.append(Rwing_all)
                        Yaw_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Roll_dFF_L_stim.append(dFF_all)
                        Roll_Outer_wing_L_stim.append(Rwing_all)
                        Roll_Inner_wing_L_stim.append(Lwing_all)
                        Roll_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Roll_dFF_R_stim.append(dFF_all)
                        Roll_Outer_wing_R_stim.append(Lwing_all)
                        Roll_Inner_wing_R_stim.append(Rwing_all)
                        Roll_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Loom_dFF_L_stim.append(dFF_all)
                        Loom_Outer_wing_L_stim.append(Rwing_all)
                        Loom_Inner_wing_L_stim.append(Lwing_all)
                        Loom_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Loom_dFF_R_stim.append(dFF_all)
                        Loom_Outer_wing_R_stim.append(Lwing_all)
                        Loom_Inner_wing_R_stim.append(Rwing_all)
                        Loom_LpRwing_R_stim.append(LpRwing_all)
                print(patterns_to_plot[stim_keys] + ': ' + str(len(LpRwing_all)))
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if len(Yaw_dFF_L_stim) == 0 or len(Yaw_dFF_R_stim) == 0:
                        continue
                    else:
                        Yaw_dFF_all = [np.vstack(Yaw_dFF_L_stim), np.vstack(Yaw_dFF_R_stim)]
                        Yaw_dFF_all = np.vstack(Yaw_dFF_all)
                        Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                        Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                        Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                        Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                        Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                        Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                        dFF_mean = np.nanmean(Yaw_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Yaw_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Yaw_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Yaw\n ', fontsize=axis_label_fontsize)
                        axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                        axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                        axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                        axs[dff_row, stim_counter].spines['left'].set_visible(False)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if len(Roll_dFF_L_stim) == 0 or len(Roll_dFF_R_stim) == 0:
                        continue
                    else:
                        Roll_dFF_all = [np.vstack(Roll_dFF_L_stim), np.vstack(Roll_dFF_R_stim)]
                        Roll_dFF_all = np.vstack(Roll_dFF_all)
                        Roll_Outer_wing_all = [np.vstack(Roll_Outer_wing_L_stim), np.vstack(Roll_Outer_wing_R_stim)]
                        Roll_Outer_wing_all = np.vstack(Roll_Outer_wing_all)
                        Roll_Inner_wing_all = [np.vstack(Roll_Inner_wing_L_stim), np.vstack(Roll_Inner_wing_R_stim)]
                        Roll_Inner_wing_all = np.vstack(Roll_Inner_wing_all)
                        Roll_LpRwing_all = [np.vstack(Roll_LpRwing_L_stim), np.vstack(Roll_LpRwing_R_stim)]
                        Roll_LpRwing_all = np.vstack(Roll_LpRwing_all)
                        dFF_mean = np.nanmean(Roll_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Roll_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Roll_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Roll_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Roll_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Roll_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Roll_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Roll_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Roll_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Roll\n ', fontsize=axis_label_fontsize)
                        axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                        axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                        axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
                        axs[dff_row, stim_counter].spines['left'].set_visible(False)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if len(Loom_dFF_L_stim) == 0 or len(Loom_dFF_R_stim) == 0:
                        continue
                    else:
                        if include_loom_trace == True:
                            stimulus_seconds = _settings['loom_duration']
                        Loom_dFF_all = [np.vstack(Loom_dFF_L_stim), np.vstack(Loom_dFF_R_stim)]
                        Loom_dFF_all = np.vstack(Loom_dFF_all)
                        Loom_Outer_wing_all = [np.vstack(Loom_Outer_wing_L_stim), np.vstack(Loom_Outer_wing_R_stim)]
                        Loom_Outer_wing_all = np.vstack(Loom_Outer_wing_all)
                        Loom_Inner_wing_all = [np.vstack(Loom_Inner_wing_L_stim), np.vstack(Loom_Inner_wing_R_stim)]
                        Loom_Inner_wing_all = np.vstack(Loom_Inner_wing_all)
                        Loom_LpRwing_all = [np.vstack(Loom_LpRwing_L_stim), np.vstack(Loom_LpRwing_R_stim)]
                        Loom_LpRwing_all = np.vstack(Loom_LpRwing_all)
                        dFF_mean = np.nanmean(Loom_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Loom_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Loom_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Loom_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ' combined n:' + str(len(Loom_LpRwing_all)))
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if plot_bs_CI == True:
                            CI_dFF = create_CI_timetrace(np.vstack(Loom_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Loom_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Loom_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Loom_LpRwing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                        if include_loom_trace == True:
                            loom_radians = np.loadtxt(loom_path(), delimiter=',')
                            loom_deg = loom_radians * (180 / np.pi)
                            loom_deg[loom_deg < 0] = 0
                            loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                            loom_with_post_pre = np.concatenate((np.full(int(seconds_pre * _settings['loom_hz']), np.nan), loom_with_post))
                            t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                            axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                            axs[loom_trace_row, stim_counter].set_ylim([0, 90])
                            axs[loom_trace_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
                            axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                            axs[loom_trace_row, stim_counter].set_ylabel('90 deg', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, va='bottom')
                            axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                            axs[loom_trace_row, stim_counter].set_yticks([])
                        axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                else:
                    axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                    axs[loom_trace_row, stim_counter].set_yticks([])
            axs[wing_row, stim_counter].set_ylim([-20, 12])
            axs[LRwing_row, stim_counter].set_ylim([-25, 15])
            axs[dff_row, stim_counter].set_ylim([-0.75, 3])
            axs[wing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[LRwing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[dff_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, int(stimulus_seconds + seconds_post)])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[rows_stim - 1, stim_counter].set_xlabel('time (s)', fontsize=axis_label_fontsize, va='center')
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3.0))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[wing_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            for i in range(rows_stim):
                axs[i, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[i, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[wing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    stimulus_seconds = 3
    all_dFF_df, all_stim_df = load_combined(datasets['data_trans_df_dir'], line)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stimulus_seconds)
    for stim_idx, stim_keys in enumerate(patterns_to_plot_trans.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_bouts_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values >= flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_bouts_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                dFF_fly.append(dFF_int - baseline_dFF_int)
                                Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                                Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                                LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(patterns_to_plot_trans[stim_keys] + ': ' + str(len(LpRwing_all)))
            axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
            axs[dff_row, stim_counter].plot(t_int, dFF_mean, color=_settings['dFF_hexcode'], linewidth=line_width)
            axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
            axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
            axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
            axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
            axs[dff_row, stim_counter].set_yticks([])
            axs[wing_row, stim_counter].set_yticks([])
            axs[LRwing_row, stim_counter].set_yticks([])
            axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds + seconds_post)])
            axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
            axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[wing_row, stim_counter].spines['top'].set_visible(False)
            axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
            axs[wing_row, stim_counter].spines['right'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
            axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['top'].set_visible(False)
            axs[dff_row, stim_counter].spines['right'].set_visible(False)
            axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
            axs[rows_stim - 1, stim_counter].spines.bottom.set_bounds((0, stimulus_seconds))
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 1, 2, 3])
                axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[LRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[rows_stim - 1, stim_counter].set_xlabel('time (s)', fontsize=axis_ticklabel_fontsize, ha='center')
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 3))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                for i in range(rows_stim - 1):
                    axs[i, stim_counter].set_xticks([])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[wing_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
            axs[wing_row, stim_counter].set_ylim([-20, 12])
            axs[LRwing_row, stim_counter].set_ylim([-25, 15])
            axs[dff_row, stim_counter].set_ylim([-0.75, 3])
            if stimulus_seconds > 2:
                axs[wing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[LRwing_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[dff_row, stim_counter].set_xlim([0 - seconds_pre * 1.5, stimulus_seconds + seconds_post])
                axs[rows_stim - 1, stim_counter].set_xticks([0, int(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
            for i in range(rows_stim):
                axs[i, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[i, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            if include_loom_trace == True:
                axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                axs[loom_trace_row, stim_counter].set_yticks([])
            if include_loom_trace == False:
                axs[0, 0].spines['left'].set_visible(True)
            for i in range(rows_stim - 1):
                axs[i, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 2 / 3, row_height)
        fig.subplots_adjust(left=0.09, bottom=0.11, right=0.974, top=0.9, wspace=0.102, hspace=0.23)
    else:
        fig.set_size_inches(_settings['column_width_3'], row_height)
        fig.subplots_adjust(wspace=0.02, hspace=0.22)
    return dict(figure=fig, results={})

def plot_entire_trial_or_snippet(datasets, pattern_dict, single_roi='True', frames_drop=5, plot_flight_stops_as_nan=False, flying_threshold=0.14, dff_row=0, wing_row=1, line='1070', date='230405', trial_num='03', fly_id='1', start=None, stop=None, scale_y=1.0, bwfilter_dFF=False, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, width=4.803149606666667):
    """
    plot entire trial of single fly's trial for 1 roi or 2 roi's 
    frames drop: drop first/last x frames (usually a little jitter at the start of the trial)
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial_num) & (all_stim_df['date'] == date)]
    dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial_num) & (all_dFF_df['date'] == date)]
    rows = 2
    columns = 1
    fig, axs = plt.subplots(rows, columns)
    if plot_flight_stops_as_nan == True:
        kinefly_vals = stim_df['aux_kinefly'].copy().values[frames_drop:-frames_drop]
        LW = stim_df['LW_filt'].copy().values[frames_drop:-frames_drop]
        RW = stim_df['RW_filt'].copy().values[frames_drop:-frames_drop]
        LmRW = stim_df['L-R_filt'].copy().values[frames_drop:-frames_drop]
        LpRW = stim_df['L+R_filt'].copy().values[frames_drop:-frames_drop]
        LW[kinefly_vals < flying_threshold] = np.nan
        RW[kinefly_vals < flying_threshold] = np.nan
        LmRW[kinefly_vals < flying_threshold] = np.nan
        LpRW[kinefly_vals < flying_threshold] = np.nan
        time_array = stim_df['timeVec'][frames_drop:-frames_drop]
        mask = (time_array >= start) & (time_array <= stop)
        filtered_time = time_array[mask]
        filtered_LW = LW[mask]
        filtered_RW = RW[mask]
        filtered_LmRW = LmRW[mask]
        filtered_LpRW = LpRW[mask]
        dff_time_array = dFF_df['frameTimes'][frames_drop:-frames_drop]
        mask_dff = (dff_time_array >= start) & (dff_time_array <= stop)
        filtered_dff_time = dff_time_array[mask_dff]
        if single_roi == True:
            if bwfilter_dFF == True:
                filt_sig = filtdFF(dFF_df['dFF'], dFF_df['imRateHz'].values[0])
                filtered_dFF = filt_sig[frames_drop:-frames_drop][mask_dff]
            else:
                filtered_dFF = dFF_df['dFF'][frames_drop:-frames_drop][mask_dff]
        elif bwfilter_dFF == True:
            filt_sig_L = filtdFF(dFF_df['dFF_left'], dFF_df['imRateHz'].values[0])
            filtered_dFF_left = filt_sig_L[frames_drop:-frames_drop][mask_dff]
            filt_sig_R = filtdFF(dFF_df['dFF_right'], dFF_df['imRateHz'].values[0])
            filtered_dFF_right = filt_sig_R[frames_drop:-frames_drop][mask_dff]
        else:
            filtered_dFF_left = dFF_df['dFF_left'][frames_drop:-frames_drop][mask_dff]
            filtered_dFF_right = dFF_df['dFF_right'][frames_drop:-frames_drop][mask_dff]
        axs[wing_row].plot(filtered_time, filtered_LW, color=_settings['lwing_hexcode'], label='left wing', linewidth=line_width)
        axs[wing_row].plot(filtered_time, filtered_RW, color=_settings['rwing_hexcode'], label='right wing', linewidth=line_width)
        if single_roi == True:
            axs[dff_row].plot(filtered_dff_time, filtered_dFF, color=_settings['dFF_hexcode'], alpha=0.75, linewidth=line_width)
        else:
            axs[dff_row].plot(filtered_dff_time, filtered_dFF_left, color=_settings['lwing_hexcode'], alpha=0.75, linewidth=line_width)
            axs[dff_row].plot(filtered_dff_time, filtered_dFF_right, color=_settings['rwing_hexcode'], alpha=0.75, linewidth=line_width)
    else:
        axs[wing_row].plot(stim_df['timeVec'][frames_drop:-frames_drop], stim_df['LW_filt'][frames_drop:-frames_drop], color=_settings['lwing_hexcode'], label='left wing', linewidth=line_width)
        axs[wing_row].plot(stim_df['timeVec'][frames_drop:-frames_drop], stim_df['RW_filt'][frames_drop:-frames_drop], color=_settings['rwing_hexcode'], label='right wing', linewidth=line_width)
        if single_roi == True:
            axs[dff_row].plot(dFF_df['frameTimes'][frames_drop:-frames_drop], dFF_df['dFF'][frames_drop:-frames_drop], color=_settings['dFF_hexcode'], alpha=0.75, linewidth=line_width)
        else:
            axs[dff_row].plot(dFF_df['frameTimes'][frames_drop:-frames_drop], dFF_df['dFF_left'][frames_drop:-frames_drop], color=_settings['lwing_hexcode'], alpha=0.5, linewidth=line_width)
            axs[dff_row].plot(dFF_df['frameTimes'][frames_drop:-frames_drop], dFF_df['dFF_right'][frames_drop:-frames_drop], color=_settings['rwing_hexcode'], alpha=0.5, linewidth=line_width)
    for stim_keys in pattern_dict.keys():
        if stim_keys == 0.0 or stim_keys == 9.5 or stim_keys == 9.99:
            continue
        elif stim_keys >= 7.7:
            start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=False)
        else:
            start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.25, half_volt=True)
        if single_roi == True:
            text_loc = 1.0
        else:
            text_loc = 6.0
        if len(start_times) != 0 and len(end_times) != 0:
            for row in range(0, rows):
                for t in range(len(start_times)):
                    t_fill = np.array([start_times[t], end_times[t]])
                    axs[row].fill_between(t_fill, -30, 150, color=_settings['visual_stim_hexcode'], alpha=_settings['alpha_patch_all'])
                    if row == 0:
                        if start == None:
                            axs[row].text(np.mean([start_times[t], end_times[t]]) - 0.6, 1.2, pattern_dict[stim_keys], fontsize=axis_ticklabel_fontsize)
                        elif start != None and start_times[t] >= start and (end_times[t] <= stop):
                            axs[row].text(np.mean([start_times[t], end_times[t]]) - 0.6, text_loc, pattern_dict[stim_keys], fontsize=axis_ticklabel_fontsize)
    if start != None and stop != None:
        axs[wing_row].set_xlim([start - 1, stop])
        axs[dff_row].set_xlim([start - 1, stop])
    axs[wing_row].set_ylim([20, 90])
    axs[wing_row].set_yticks([25, 50, 75])
    axs[wing_row].set_ylabel('Wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, ha='center')
    axs[wing_row].spines.left.set_bounds((25, 75))
    axs[dff_row].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center')
    if single_roi == True:
        axs[dff_row].set_ylim([-0.2, scale_y + 0.2])
        axs[dff_row].set_yticks([0, scale_y])
        axs[dff_row].spines.left.set_bounds((0, scale_y))
    else:
        axs[dff_row].set_ylim([-1, 6])
        axs[dff_row].set_yticks([0, 2, 4])
        axs[dff_row].spines.left.set_bounds((0, 4))
    for r in range(rows):
        axs[r].spines['right'].set_visible(False)
        axs[r].spines['top'].set_visible(False)
        axs[r].spines.bottom.set_bounds((start, stop))
        axs[r].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[r].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs[r].spines.left.set_linewidth(_settings['line_width_spines'])
        axs[r].spines.bottom.set_linewidth(_settings['line_width_spines'])
        axs[r].set_yticklabels(axs[r].get_yticks(), fontsize=axis_ticklabel_fontsize)
        if r == 0:
            axs[r].set_xticks([])
            axs[r].spines['bottom'].set_visible(False)
        else:
            axs[r].set_xlabel('time (s)', fontsize=axis_label_fontsize)
            axs[r].spines['bottom'].set_visible(True)
            if start != None and stop != None:
                axs[r].spines.bottom.set_bounds((start, stop))
                steps = np.linspace(start, stop, int((stop - start) / 10) + 1)
                axs[r].set_xticks([int(step) for step in steps])
            axs[r].set_xticklabels(axs[r].get_xticks(), fontsize=axis_ticklabel_fontsize)
    fig.align_ylabels(axs[:])
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(width, _settings['column_width_2'] / 4)
        if single_roi:
            fig.subplots_adjust(left=0.105, bottom=0.16, right=0.973, top=0.96, wspace=0.2, hspace=0.212)
        else:
            fig.subplots_adjust(left=0.135, bottom=0.16, right=0.973, top=0.921, wspace=0.2, hspace=0.212)
    else:
        fig.set_size_inches(_settings['column_width_3'] * (2 / 3), _settings['column_width_3'] / 4)
        fig.subplots_adjust(wspace=0.2, hspace=0.151, top=0.917)
    return dict(figure=fig, results={})

def plot_avg_trace_per_visual_stimulus_trans_exp_overlayed_simplified(datasets, pattern_dict_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, seconds_post_plot=0.5, airpuff_threshold=2.0, plot_individual_traces=True, plot_fly_means=True, plot_bs_CI=True, stim_seconds=3, air_puff_pre_s=1, colormap='YlGnBu', pitch_color='black', CDOU_color='#4632a8', alpha_patch=0.5, alpha_CI=0.6, include_visual_patch=True, line_width=1, axis_ticklabel_fontsize=6, axis_label_fontsize=6):
    """
    *less traces, thinner, no legend
    generate avg trace plots for all data combined overlayed 1 plot; just dff
    single roi for dff
    still collect L+R information in case want to add in later
    mean of fly means 
    bs CI  
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    rows_stim = 1
    fig, axs = plt.subplots(rows_stim, 1)
    stim_counter = 0
    if colormap != 'black':
        cmap = plt.get_cmap(colormap, 256)
        point_n_array = np.arange(len(pattern_dict_to_plot))
        point_color_vals = np.round(np.linspace(0.4, 0.9, len(pattern_dict_to_plot)), 2)
        col_dict = dict(zip(point_n_array, point_color_vals))
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stim_seconds)
    if include_visual_patch == True:
        axs.fill_between([0, stim_seconds], -0.5, 3.0, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
    for stim_idx, stim_keys in enumerate(pattern_dict_to_plot.keys()):
        if pattern_dict_to_plot[stim_keys] == 'Pitch\ndown':
            color = pitch_color
        elif pattern_dict_to_plot[stim_keys] == 'Center down\nouter up':
            color = CDOU_color
        elif colormap == 'black':
            color = 'black'
        else:
            color = cmap(col_dict[stim_idx])
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                dFF_fly.append(dFF_int - baseline_dFF_int)
                                Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                                Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                                LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
                                if plot_individual_traces == True:
                                    axs.plot(t_int, dFF_int - baseline_dFF_int, color=color, alpha=0.025, linewidth=line_width)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
                if plot_fly_means == True:
                    axs.plot(t_int, dFF_fly_mean, color=color, alpha=0.2)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print('stim: ' + str(pattern_dict_to_plot[stim_keys]) + ' fly n=' + str(len(dFF_all)))
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs.fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['bs_CI_hexcode'], alpha=alpha_CI, edgecolor='none')
            axs.plot(t_int, dFF_mean, color=color, label=pattern_dict_to_plot[stim_keys], linewidth=line_width)
            axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center', labelpad=0)
            axs.set_yticks([0, 1, 2, 3])
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.set_ylim([-0.5, 3])
            axs.set_xlim([-seconds_pre * 1.5, stim_seconds + seconds_post_plot])
            axs.set_xticks([0, stim_seconds])
            axs.spines.bottom.set_bounds((0, stim_seconds))
            axs.spines.left.set_bounds((0, 3.0))
            axs.set_xlabel('time (s)', fontsize=axis_label_fontsize, ha='center', labelpad=-1)
            axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs.spines.left.set_linewidth(_settings['line_width_spines'])
            axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
            axs.set_xticklabels(axs.get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs.set_yticklabels(axs.get_yticks(), fontsize=axis_ticklabel_fontsize)
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.18, _settings['column_width_2'] * 0.18)
        fig.subplots_adjust(left=0.179, bottom=0.104, right=0.946, top=0.963)
    else:
        fig.set_size_inches(_settings['column_width_3'] / 2, _settings['column_width_3'] / 4)
        fig.subplots_adjust(left=0.179, bottom=0.187, right=0.751, top=0.821)
    return dict(figure=fig, results={})

def plot_updown_fullspeed_overlayed(datasets, pattern_dict_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=0.5, seconds_post=1, seconds_post_plot=0.5, airpuff_threshold=2.0, plot_individual_traces=True, plot_fly_means=True, plot_bs_CI=True, stim_seconds=3, air_puff_pre_s=1, colormap='black', alpha_patch=0.5, alpha_CI=0.6, include_visual_patch=True, line_width=1, axis_ticklabel_fontsize=6, axis_label_fontsize=6):
    """
    generate avg trace plots for all data combined overlayed 1 plot; just dff
    single roi for dff
    still collect L+R information in case want to add in later
    mean of fly means 
    bs CI  
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    rows_stim = 1
    fig, axs = plt.subplots(rows_stim, 1)
    stim_counter = 0
    if colormap != 'black':
        cmap = plt.get_cmap(colormap, 256)
        point_n_array = np.arange(len(pattern_dict_to_plot))
        point_color_vals = np.round(np.linspace(0.4, 0.9, len(pattern_dict_to_plot)), 2)
        col_dict = dict(zip(point_n_array, point_color_vals))
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stim_seconds)
    if include_visual_patch == True:
        axs.fill_between([0, stim_seconds], -0.5, 3.0, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
    for stim_idx, stim_keys in enumerate(pattern_dict_to_plot.keys()):
        if colormap == 'black':
            color = 'black'
        else:
            color = cmap(col_dict[stim_idx])
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        mean_all_flies_entire_stim = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.2, half_volt=False)
                if len(start_times) != 0 and len(end_times) != 0:
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            dFF_train_int = np.concatenate((pre_dFF, stim_dFF, post_dFF))
                            dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                            baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            dFF_fly.append(dFF_int - baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
                            if plot_individual_traces == True:
                                axs.plot(t_int, dFF_int - baseline_dFF_int, color=color, alpha=0.025, linewidth=line_width)
            if len(dFF_fly) != 0:
                dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
                mean_per_fly_entire_stim = np.nanmean(np.array(dFF_fly)[:, frames_pre:frames_pre + frames_stim])
                mean_all_flies_entire_stim.append(mean_per_fly_entire_stim)
                if plot_fly_means == True:
                    axs.plot(t_int, dFF_fly_mean, color=color, alpha=0.2)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(f"{' '.join(pattern_dict_to_plot[stim_keys].split())} N={len(dFF_all)}")
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs.fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['bs_CI_hexcode'], alpha=alpha_CI, edgecolor='none')
            axs.plot(t_int, dFF_mean, color=color, label=pattern_dict_to_plot[stim_keys], linewidth=line_width)
            grand_mean_for_stim_period = np.nanmean(mean_all_flies_entire_stim)
            # Keep the original scalar CI draws even though only N is reported.
            grand_mean_CI = create_CI(mean_all_flies_entire_stim, iterations=1000, CI=95.0)
            axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center', labelpad=0)
            axs.set_yticks([-0.5, 0, 0.5, 1, 2, 3])
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.set_ylim([-0.5, 3])
            axs.set_xlim([-seconds_pre * 1.5, stim_seconds + seconds_post_plot])
            axs.set_xticks([0, stim_seconds])
            axs.spines.bottom.set_bounds((0, stim_seconds))
            axs.spines.left.set_bounds((-0.5, 3.0))
            axs.set_xlabel('time (s)', fontsize=axis_label_fontsize, ha='center', labelpad=-1)
            axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs.spines.left.set_linewidth(_settings['line_width_spines'])
            axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
            axs.set_xticklabels(axs.get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs.set_yticklabels(axs.get_yticks(), fontsize=axis_ticklabel_fontsize)
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * 0.25, _settings['column_width_2'] * 0.15)
        fig.subplots_adjust(left=0.373, bottom=0.173, right=0.843, top=0.964)
    else:
        fig.set_size_inches(_settings['column_width_3'] / 2, _settings['column_width_3'] / 4)
        fig.subplots_adjust(left=0.3, bottom=0.165, right=0.706, top=0.92)
    return dict(figure=fig, results={})

def plot_peak_dff_by_visual_stim_panel_width_shade_box_CI(datasets, pattern_dict, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, plot_fly_means=True, plot_bs_CI=True, stim_seconds=3, air_puff_pre_s=1, response_curve_to_compare=-1, stat_method='mean', linewidth=0.25, jitter_means=True):
    """
    plot peak dff per stimuli and error bars
    single roi for dff
    mean of fly means 
    bs CI  
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    response curve to compare: -1 take last second of stimulus, -2 last 2, -3 entire stimuli
    stat_method: take mean, median, or max of specified time period
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    fig, axs = plt.subplots(1, 1)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stim_seconds)
    frames_for_avg = int(imRateHz_mean * response_curve_to_compare)
    for stim_idx, stim_keys in enumerate(pattern_dict.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end - start, frames_pre + frames_stim)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                stim_dFF_int = dFF_int[frames_pre:frames_pre + frames_stim]
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Lstim_wing_int = Lwing_int[frames_pre:frames_pre + frames_stim]
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                Rstim_wing_int = Rwing_int[frames_pre:frames_pre + frames_stim]
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                LpRstim_wing_int = LpRwing_int[frames_pre:frames_pre + frames_stim]
                                if stat_method == 'mean':
                                    dFF_fly.append(np.nanmean((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmean((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmean((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmean((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'median':
                                    dFF_fly.append(np.nanmedian((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmedian((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmedian((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmedian((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'max':
                                    dFF_fly.append(np.nanmax((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmax((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmax((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmax((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
            if len(dFF_fly) != 0:
                if stat_method == 'median':
                    dFF_fly_mean = np.nanmedian(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmedian(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmedian(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmedian(LpRwing_fly, axis=0)
                else:
                    dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
                if plot_fly_means == True:
                    if jitter_means == True:
                        jitter = np.random.uniform(-0.1, 0.1, 1)
                        data_points = [dFF_fly_mean]
                        axs.scatter(np.full_like(data_points, stim_idx) + jitter, data_points, color='black', alpha=1.0, s=_settings['point_size_min'], zorder=3, edgecolor='none', marker='o')
                        axs.scatter(pattern_dict[stim_keys], dFF_fly_mean, color='black', alpha=0.0, s=3)
                    else:
                        axs.scatter(pattern_dict[stim_keys], dFF_fly_mean, alpha=1.0, color='black', s=_settings['point_size_min'], marker='o', edgecolors='none')
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(f'{pattern_dict[stim_keys]} n={len(dFF_all)}')
            if plot_bs_CI == True:
                dFF_CI = create_CI(dFF_all, CI=95.0, iterations=1000)
                Lwing_CI = create_CI(Lwing_all, CI=95.0, iterations=1000)
                Rwing_CI = create_CI(Rwing_all, CI=95.0, iterations=1000)
                LpRwing_CI = create_CI(LpRwing_all, CI=95.0, iterations=1000)
                ci_height = dFF_CI[1] - dFF_CI[0]
                rect = Rectangle((stim_idx - 0.25, dFF_CI[0]), width=0.5, height=ci_height, facecolor=_settings['bs_CI_hexcode'], alpha=0.5, edgecolor='none', zorder=1)
                axs.add_patch(rect)
                axs.hlines(dFF_mean, stim_idx - 0.25, stim_idx + 0.25, color='blue', lw=linewidth, zorder=2)
            else:
                axs.scatter(pattern_dict[stim_keys], dFF_mean, color='black', s=_settings['point_size_min'])
    axs.spines.bottom.set_bounds((0, len(pattern_dict.keys()) - 1))
    axs.spines.left.set_bounds((-0.5, 5.0))
    axs.set_xlim([-0.5, len(pattern_dict.keys()) - 0.5])
    axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=_settings['axes_title_font'], ha='center')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    if plot_fly_means == True:
        axs.set_ylim([-0.5, 5.2])
    else:
        axs.set_ylim([-0.5, 5.0])
    axs.set_yticks([-0.5, 0, 2.5, 5])
    axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    axs.spines.left.set_linewidth(_settings['line_width_spines'])
    axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=_settings['ticklabel_font'], rotation=90)
    axs.set_yticklabels(axs.get_yticklabels(), fontsize=_settings['ticklabel_font'])
    fig.tight_layout()
    fig.set_size_inches(_settings['column_width_2'] * 0.2, _settings['column_width_2'] * 0.22)
    fig.subplots_adjust(left=0.125, bottom=0.26, right=0.98, top=0.969)
    return dict(figure=fig, results={})

def plot_peak_dff_by_visual_stim_panel_width_shade_box_CI_stats(datasets, pattern_dict, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, plot_fly_means=True, plot_bs_CI=True, stim_seconds=3, air_puff_pre_s=1, response_curve_to_compare=-1, stat_method='mean', linewidth=0.25, jitter_means=True, *, plot=True):
    """
    plot peak dff per stimuli and error bars and report statistics
    single roi for dff
    mean of fly means 
    bs CI  
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    response curve to compare: -1 take last second of stimulus, -2 last 2, -3 entire stimuli
    stat_method: take mean, median, or max of specified time period
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if plot:
        fig, axs = plt.subplots(1, 1)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stim_seconds)
    frames_for_avg = int(imRateHz_mean * response_curve_to_compare)
    stim_mean_dict = {}
    fly_ids_list = all_dFF_df['fly_id'].unique()
    for stim_idx, stim_keys in enumerate(pattern_dict.keys()):
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        fly_ids = []
        for fly_id in fly_ids_list:
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end - start, frames_pre + frames_stim)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                stim_dFF_int = dFF_int[frames_pre:frames_pre + frames_stim]
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Lstim_wing_int = Lwing_int[frames_pre:frames_pre + frames_stim]
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                Rstim_wing_int = Rwing_int[frames_pre:frames_pre + frames_stim]
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                LpRstim_wing_int = LpRwing_int[frames_pre:frames_pre + frames_stim]
                                if stat_method == 'mean':
                                    dFF_fly.append(np.nanmean((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmean((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmean((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmean((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'median':
                                    dFF_fly.append(np.nanmedian((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmedian((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmedian((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmedian((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'max':
                                    dFF_fly.append(np.nanmax((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmax((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmax((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmax((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
            if len(dFF_fly) != 0:
                fly_ids.append(fly_id)
                if stat_method == 'median':
                    dFF_fly_mean = np.nanmedian(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmedian(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmedian(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmedian(LpRwing_fly, axis=0)
                else:
                    dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
                    Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                    Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                    LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
                if plot_fly_means == True:
                    if jitter_means == True:
                        jitter = np.random.uniform(-0.1, 0.1, 1)
                        data_points = [dFF_fly_mean]
                        if plot:
                            axs.scatter(np.full_like(data_points, stim_idx) + jitter, data_points, color='black', alpha=1.0, s=_settings['point_size_min'], zorder=3, edgecolor='none', marker='o')
                        if plot:
                            axs.scatter(pattern_dict[stim_keys], dFF_fly_mean, color='black', alpha=0.0, s=3)
                    elif plot:
                        axs.scatter(pattern_dict[stim_keys], dFF_fly_mean, alpha=1.0, color='black', s=_settings['point_size_min'], marker='o', edgecolors='none')
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(f'{pattern_dict[stim_keys]} n={len(dFF_all)}')
            if plot_bs_CI == True:
                dFF_CI = create_CI(dFF_all, CI=95.0, iterations=1000)
                Lwing_CI = create_CI(Lwing_all, CI=95.0, iterations=1000)
                Rwing_CI = create_CI(Rwing_all, CI=95.0, iterations=1000)
                LpRwing_CI = create_CI(LpRwing_all, CI=95.0, iterations=1000)
                ci_height = dFF_CI[1] - dFF_CI[0]
                if plot:
                    rect = Rectangle((stim_idx - 0.25, dFF_CI[0]), width=0.5, height=ci_height, facecolor=_settings['bs_CI_hexcode'], alpha=0.5, edgecolor='none', zorder=1)
                if plot:
                    axs.add_patch(rect)
                if plot:
                    axs.hlines(dFF_mean, stim_idx - 0.25, stim_idx + 0.25, color='blue', lw=linewidth, zorder=2)
            elif plot:
                axs.scatter(pattern_dict[stim_keys], dFF_mean, color='black', s=_settings['point_size_min'])
            stim_mean_dict[pattern_dict[stim_keys]] = dFF_all
    print(f'{list(stim_mean_dict.keys())[0]} x {list(stim_mean_dict.keys())[1]}')
    args = [list(v) for v in stim_mean_dict.values()]
    print('permutation tests')
    perm_p_vals = []
    perm_p_vals.append(permutation_test(args[0], args[1], size=1000000))
    print(f'permutation p val: {perm_p_vals[0]:.3e}')
    if plot:
        axs.spines.bottom.set_bounds((0, len(pattern_dict.keys()) - 1))
    if plot:
        axs.spines.left.set_bounds((-0.5, 5.0))
    if plot:
        axs.set_xlim([-0.5, len(pattern_dict.keys()) - 0.5])
    if plot:
        axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=_settings['axes_title_font'], ha='center')
    if plot:
        axs.spines['right'].set_visible(False)
    if plot:
        axs.spines['top'].set_visible(False)
    if plot_fly_means == True:
        if plot:
            axs.set_ylim([-0.5, 5.2])
    elif plot:
        axs.set_ylim([-0.5, 5.0])
    if plot:
        axs.set_yticks([-0.5, 0, 2.5, 5])
    if plot:
        axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    if plot:
        axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    if plot:
        axs.spines.left.set_linewidth(_settings['line_width_spines'])
    if plot:
        axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
    if plot:
        axs.set_xticklabels(axs.get_xticklabels(), fontsize=_settings['ticklabel_font'], rotation=90)
    if plot:
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=_settings['ticklabel_font'])
    if plot:
        fig.tight_layout()
    if plot:
        fig.set_size_inches(_settings['column_width_2'] * 0.1, _settings['column_width_2'] * 0.22)
    if plot:
        fig.subplots_adjust(left=0.125, bottom=0.26, right=0.98, top=0.969)
    return dict(figure=fig if plot else None, results={'stim_mean_dict': stim_mean_dict, 'perm_p_vals': perm_p_vals})

def plot_peak_dff_by_visual_stim_panel_width_regression(datasets, pattern_dict, volt_int_dict, regression_type, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, plot_fly_regressions=True, stim_seconds=3, air_puff_pre_s=1, response_curve_to_compare=-1, stat_method='mean', linewidth=0.25, jitter_means=True, width_deg_or_percent='percent', *, plot=True):
    """
    plot peak dff per stimuli (36-216) and report ORD regression slope
    single roi for dff
    mean of fly means 
    bs CI  
    volt_int_dict: voltage to int value of stimulus width
    regression type: can be 'ODR' or 'linregress' *NOTE: there is no variation in x (stimulus width) so ODR overkill
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    response curve to compare: -1 take last second of stimulus, -2 last 2, -3 entire stimuli
    stat_method: take mean, median, or max of specified time period
    width_deg_or_percent = x units reported in degrees azimuth or % of azimuth (x/360)
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    if plot:
        fig, axs = plt.subplots(1, 1)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * stim_seconds)
    frames_for_avg = int(imRateHz_mean * response_curve_to_compare)
    slopes_all_flies = []
    offset_all_flies = []
    r_squared_all_flies = []
    for fly_id in all_dFF_df['fly_id'].unique():
        X = []
        Y = []
        for stim_idx, stim_keys in enumerate(pattern_dict.keys()):
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if len(stim_df) > 3000:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=True)
                    if len(start_times) != 0 and len(end_times) != 0:
                        for ind, start in enumerate(start_times):
                            start_t = start - seconds_pre
                            end = end_times[ind]
                            end_t = end + seconds_post
                            pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                            pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                            stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF'].values
                            stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                            post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                            post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                            pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                            stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                            post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                            pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                            stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                            post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                            pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                            stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                            post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                            pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                            stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                            post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                            flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                            air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                            if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                                t_int = np.linspace(start_t - start, end - start, frames_pre + frames_stim)
                                t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                                dFF_train_int = np.concatenate((pre_dFF, stim_dFF))
                                dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                                baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                                stim_dFF_int = dFF_int[frames_pre:frames_pre + frames_stim]
                                t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t)) - start
                                Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing))
                                Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                                Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                                Lstim_wing_int = Lwing_int[frames_pre:frames_pre + frames_stim]
                                Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing))
                                Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                                Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                                Rstim_wing_int = Rwing_int[frames_pre:frames_pre + frames_stim]
                                LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing))
                                LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                                LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                                LpRstim_wing_int = LpRwing_int[frames_pre:frames_pre + frames_stim]
                                if stat_method == 'mean':
                                    dFF_fly.append(np.nanmean((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmean((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmean((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmean((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'median':
                                    dFF_fly.append(np.nanmedian((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmedian((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmedian((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmedian((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
                                elif stat_method == 'max':
                                    dFF_fly.append(np.nanmax((stim_dFF_int - baseline_dFF_int)[frames_for_avg:]))
                                    Lwing_fly.append(np.nanmax((Lstim_wing_int - Lbaseline_wing_int)[frames_for_avg:]))
                                    Rwing_fly.append(np.nanmax((Rstim_wing_int - Rbaseline_wing_int)[frames_for_avg:]))
                                    LpRwing_fly.append(np.nanmax((LpRstim_wing_int - LpRbaseline_wing_int)[frames_for_avg:]))
            if len(dFF_fly) != 0:
                X.append([volt_int_dict[stim_keys]] * len(dFF_fly))
                Y.append(dFF_fly)
                if width_deg_or_percent == 'deg':
                    jitter_val = 5
                else:
                    jitter_val = 0.01
                jitter = np.random.uniform(-jitter_val, jitter_val, len(dFF_fly))
                data_points = [dFF_fly]
                if plot:
                    axs.scatter(np.full_like(data_points, volt_int_dict[stim_keys]) + jitter, data_points, color='black', alpha=1.0, s=_settings['point_size_min'], zorder=3, edgecolor='none', marker='o')
        x_arr = np.array([x for sub in X for x in sub])
        y_arr = np.array([y for sub in Y for y in sub])
        x_new = x_arr[~np.isnan(y_arr)]
        y_new = y_arr[~np.isnan(y_arr)]
        if regression_type == 'ODR':
            sx = 1e-06
            sy = np.nanstd(y_new)
            output = orthogonal_distance_regression(x_new, y_new, sx, sy, beta_estimate_0=0.01, beta_estimate_1=0.5)
            slopes_all_flies.append(output.beta[0])
            offset_all_flies.append(output.beta[1])
        else:
            res = stats.linregress(x_new, y_new)
            slopes_all_flies.append(res.slope)
            offset_all_flies.append(res.intercept)
            r_squared_all_flies.append(res.rvalue ** 2)
    mean_slope = np.nanmean(slopes_all_flies)
    CI_slopes = create_CI(slopes_all_flies, CI=95.0, iterations=1000)
    mean_offset = np.nanmean(offset_all_flies)
    mean_r_squared = np.nanmean(r_squared_all_flies)
    CI_r_squared = create_CI(r_squared_all_flies, CI=95.0, iterations=1000)
    print(f'mean slope across fly regressions m={round(mean_slope, 3)} CI [{round(CI_slopes[0], 3)}, {round(CI_slopes[1], 3)}]')
    print(f'mean r**2 across fly regressions r**2={round(mean_r_squared, 3)} CI [{round(CI_r_squared[0], 3)}, {round(CI_r_squared[1], 3)}]')
    if width_deg_or_percent == 'deg':
        x_test = np.linspace(30, 220, 100)
    else:
        x_test = np.linspace(0.08, 0.62, 100)
    y_pred = mean_slope * x_test + mean_offset
    if plot:
        axs.plot(x_test, y_pred, color='r')
    if plot_fly_regressions:
        for i in range(len(slopes_all_flies)):
            y_pred = slopes_all_flies[i] * x_test + offset_all_flies[i]
            if plot:
                axs.plot(x_test, y_pred, color='r', alpha=0.2)
    if width_deg_or_percent == 'deg':
        if plot:
            axs.set_xlim([25, 225])
        if plot:
            axs.set_xlabel('Stimulus width (deg)', rotation=0, fontsize=_settings['axes_title_font'], ha='center')
    else:
        if plot:
            axs.set_xlim([0.05, 0.62])
        if plot:
            axs.set_xlabel('Stimulus width % of 360 deg', rotation=0, fontsize=_settings['axes_title_font'], ha='center')
    stim_widths = sorted(volt_int_dict.values())
    if plot:
        axs.spines.bottom.set_bounds((stim_widths[0], stim_widths[-1]))
    if plot:
        axs.set_xticks(stim_widths)
    if plot:
        axs.set_xticklabels(stim_widths, fontsize=_settings['ticklabel_font'], rotation=90)
    if plot:
        axs.spines.left.set_bounds((-0.5, 5.0))
    if plot:
        axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=_settings['axes_title_font'], ha='center')
    if plot:
        axs.spines['right'].set_visible(False)
    if plot:
        axs.spines['top'].set_visible(False)
    if plot:
        axs.set_ylim([-0.5, 5.2])
    if plot:
        axs.set_yticks([-0.5, 0, 2.5, 5])
    if plot:
        axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    if plot:
        axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    if plot:
        axs.spines.left.set_linewidth(_settings['line_width_spines'])
    if plot:
        axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
    if plot:
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=_settings['ticklabel_font'])
    if plot:
        fig.tight_layout()
    if plot:
        fig.set_size_inches(_settings['column_width_2'] * 0.3, _settings['column_width_2'] * 0.4)
    if plot:
        fig.subplots_adjust(left=0.125, bottom=0.26, right=0.98, top=0.969)
    return dict(figure=fig if plot else None, results={'mean_slope': mean_slope, 'CI_slopes': CI_slopes, 'mean_r_squared': mean_r_squared, 'CI_r_squared': CI_r_squared})

def plot_avg_trace_per_visual_stimulus_descending_branches(datasets, patterns_to_plot, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, plot_bs_CI=True, air_puff_pre_s=1, axis_label_fontsize=24, axis_ticklabel_fontsize=20, label_pad=44, line_width=2, alpha_CI=0.6, alpha_patch=0.5, dff_row=0, wing_row=1, LRwing_row=2, LmRwing_row=3, loom_trace_row=3, column_width=7.20472441, include_loom_trace=False, include_LmR=False):
    """
    generate avg trace plots for each visual stimuli from either original experiment of recording left and right branches
    plot all lateral stimuli as 'left turns' and for lateral stim, dff side is coded by ipsilateral or contralateral
    mean of fly means 
    bs CI  
    data_directory= either original experiment or trans directory 
    air_puff_pre_s: number of seconds before the "pre" period to check for air puff
    plot original experiment = True full stim panel False only translation experiments 
    *plotted as left turns for lateral stimuli
    """
    _settings = config.load('two_photon')['settings']
    if include_loom_trace == False and include_LmR == False:
        row_height = 2.5
        loom_trace = ''
        lmr_trace = ''
        rows_stim = 3
    if include_loom_trace == False and include_LmR == True:
        row_height = 3
        loom_trace = ''
        lmr_trace = '_LmR_trace_'
        rows_stim = 4
    if include_loom_trace == True and include_LmR == False:
        row_height = 3
        loom_trace = '_loom_trace'
        lmr_trace = ''
        rows_stim = 4
    if include_loom_trace == True and include_LmR == True:
        row_height = 3.5
        loom_trace = '_loom_trace'
        lmr_trace = '_LmR_trace_'
        rows_stim = 5
    columns_stim = 8
    fig, axs = plt.subplots(rows_stim, columns_stim)
    stim_counter = 0
    exp_name = 'original_experiment'
    all_dFF_df, all_stim_df = load_combined(datasets['data_general_df_dir'], line)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    Yaw_Inner_dFF_all = []
    Yaw_Outer_dFF_all = []
    Yaw_Outer_wing_all = []
    Yaw_Inner_wing_all = []
    Yaw_LpRwing_all = []
    Roll_Inner_dFF_all = []
    Roll_Outer_dFF_all = []
    Roll_Outer_wing_all = []
    Roll_Inner_wing_all = []
    Roll_LpRwing_all = []
    Loom_Inner_dFF_all = []
    Loom_Outer_dFF_all = []
    Loom_Outer_wing_all = []
    Loom_Inner_wing_all = []
    Loom_LpRwing_all = []
    Yaw_Inner_dFF_L_stim = []
    Yaw_Outer_dFF_L_stim = []
    Yaw_Outer_wing_L_stim = []
    Yaw_Inner_wing_L_stim = []
    Yaw_LpRwing_L_stim = []
    Roll_Inner_dFF_L_stim = []
    Roll_Outer_dFF_L_stim = []
    Roll_Outer_wing_L_stim = []
    Roll_Inner_wing_L_stim = []
    Roll_LpRwing_L_stim = []
    Loom_Inner_dFF_L_stim = []
    Loom_Outer_dFF_L_stim = []
    Loom_Outer_wing_L_stim = []
    Loom_Inner_wing_L_stim = []
    Loom_LpRwing_L_stim = []
    Yaw_Inner_dFF_R_stim = []
    Yaw_Outer_dFF_R_stim = []
    Yaw_Outer_wing_R_stim = []
    Yaw_Inner_wing_R_stim = []
    Yaw_LpRwing_R_stim = []
    Roll_Inner_dFF_R_stim = []
    Roll_Outer_dFF_R_stim = []
    Roll_Outer_wing_R_stim = []
    Roll_Inner_wing_R_stim = []
    Roll_LpRwing_R_stim = []
    Loom_Inner_dFF_R_stim = []
    Loom_Outer_dFF_R_stim = []
    Loom_Outer_wing_R_stim = []
    Loom_Inner_wing_R_stim = []
    Loom_LpRwing_R_stim = []
    for stim_idx, stim_keys in enumerate(patterns_to_plot.keys()):
        left_dFF_all = []
        right_dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            left_dFF_fly = []
            right_dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                if stim_keys >= 7.7:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.075, half_volt=False)
                else:
                    start_times, end_times = get_start_end_indices_stimulus_df(stim_df, stim_keys, range_volts=0.25, half_volt=True)
                if len(start_times) != 0 and len(end_times) != 0:
                    if len(left_dFF_all) == 0:
                        stimulus_seconds = round(np.mean(np.array(end_times) - np.array(start_times)))
                        frames_stim = int(imRateHz_mean * stimulus_seconds)
                    for ind, start in enumerate(start_times):
                        start_t = start - seconds_pre
                        end = end_times[ind]
                        end_t = end + seconds_post
                        left_pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_left'].values
                        right_pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_right'].values
                        pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                        left_stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF_left'].values
                        right_stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['dFF_right'].values
                        stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end)]['frameTimes'].values
                        left_post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF_left'].values
                        right_post_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['dFF_right'].values
                        post_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= end) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                        pre_left_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                        stim_left_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['LW_filt'].values
                        post_left_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                        pre_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                        stim_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['RW_filt'].values
                        post_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                        pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                        stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['L+R_filt'].values
                        post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                        pre_wing_t = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                        stim_wing_t = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end)]['timeVec'].values
                        post_wing_t = stim_df.loc[(stim_df['timeVec'] >= end) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                        flight_stops_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < end_t)]['aux_kinefly_filt'].values < flying_threshold)
                        air_puffs_num = np.sum(stim_df.loc[(stim_df['timeVec'] >= start_t - air_puff_pre_s) & (stim_df['timeVec'] < end_t)]['air_puff'].values > airpuff_threshold)
                        if len(pre_dFF_times) > 1 and len(post_dFF_times) > 1 and (flight_stops_num == 0) and (air_puffs_num == 0):
                            t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                            t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times, post_dFF_times)) - start
                            left_dFF_train_int = np.concatenate((left_pre_dFF, left_stim_dFF, left_post_dFF))
                            right_dFF_train_int = np.concatenate((right_pre_dFF, right_stim_dFF, right_post_dFF))
                            left_dFF_int = interpolate_trace(t_dFF_train_int, left_dFF_train_int, t_int)
                            left_baseline_dFF_int = np.nanmean(left_dFF_int[:frames_pre])
                            right_dFF_int = interpolate_trace(t_dFF_train_int, right_dFF_train_int, t_int)
                            right_baseline_dFF_int = np.nanmean(right_dFF_int[:frames_pre])
                            t_wing_train_int = np.concatenate((pre_wing_t, stim_wing_t, post_wing_t)) - start
                            Lwing_train_int = np.concatenate((pre_left_wing, stim_left_wing, post_left_wing))
                            Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                            Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                            Rwing_train_int = np.concatenate((pre_right_wing, stim_right_wing, post_right_wing))
                            Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                            Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                            LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                            LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                            LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                            left_dFF_fly.append(left_dFF_int - left_baseline_dFF_int)
                            right_dFF_fly.append(right_dFF_int - right_baseline_dFF_int)
                            Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                            Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                            LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            if len(left_dFF_fly) != 0 and len(right_dFF_fly) != 0:
                left_dFF_fly_mean = np.nanmean(left_dFF_fly, axis=0)
                right_dFF_fly_mean = np.nanmean(right_dFF_fly, axis=0)
                Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
                Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
                LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
                left_dFF_all.append(left_dFF_fly_mean)
                right_dFF_all.append(right_dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(left_dFF_all) != 0 and len(right_dFF_all) != 0:
            if patterns_to_plot[stim_keys].split(' ')[0] != 'Yaw' and patterns_to_plot[stim_keys].split(' ')[0] != 'Roll' and (patterns_to_plot[stim_keys].split(' ')[0] != 'Loom'):
                left_dFF_mean = np.nanmean(left_dFF_all, axis=0)
                right_dFF_mean = np.nanmean(right_dFF_all, axis=0)
                Lwing_mean = np.nanmean(Lwing_all, axis=0)
                Rwing_mean = np.nanmean(Rwing_all, axis=0)
                LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
                print(' '.join(patterns_to_plot[stim_keys].split()) + ': ' + str(len(LpRwing_all)))
                if plot_bs_CI == True:
                    CI_dFF_left = create_CI_timetrace(np.vstack(left_dFF_all), iterations=1000, CI=95.0)
                    CI_dFF_right = create_CI_timetrace(np.vstack(right_dFF_all), iterations=1000, CI=95.0)
                    CI_lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                    CI_rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    CI_lrwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                    CI_lmrwing = create_CI_timetrace(np.vstack(Lwing_all) - np.vstack(Rwing_all), iterations=1000, CI=95.0)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_lwing[:, 0], CI_lwing[:, 1], color=_settings['lwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[wing_row, stim_counter].fill_between(t_int, CI_rwing[:, 0], CI_rwing[:, 1], color=_settings['rwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_left[:, 0], CI_dFF_left[:, 1], color=_settings['left_dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_right[:, 0], CI_dFF_right[:, 1], color=_settings['right_dFF_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                    if include_LmR:
                        axs[LmRwing_row, stim_counter].fill_between(t_int, CI_lmrwing[:, 0], CI_lmrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                if include_loom_trace == True and patterns_to_plot[stim_keys] == 'Front \nloom':
                    stimulus_seconds = _settings['loom_duration']
                axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[dff_row, stim_counter].plot(t_int, left_dFF_mean, color=_settings['left_dFF_hexcode'], linewidth=line_width)
                axs[dff_row, stim_counter].plot(t_int, right_dFF_mean, color=_settings['right_dFF_hexcode'], linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Lwing_mean, color=_settings['lwing_hexcode'], label='L wing', linewidth=line_width)
                axs[wing_row, stim_counter].plot(t_int, Rwing_mean, color=_settings['rwing_hexcode'], label='R wing', linewidth=line_width)
                axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                if include_LmR:
                    axs[LmRwing_row, stim_counter].plot(t_int, Lwing_mean - Rwing_mean, color=_settings['lrwing_hexcode'], label='L-R wing', linewidth=line_width)
                    axs[LmRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                axs[0, stim_counter].set_title(patterns_to_plot[stim_keys], fontsize=axis_label_fontsize)
                if include_loom_trace == True:
                    if patterns_to_plot[stim_keys] == 'Front \nloom':
                        loom_radians = np.loadtxt(loom_path(), delimiter=',')
                        loom_deg = loom_radians * (180 / np.pi)
                        loom_deg[loom_deg < 0] = 0
                        loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                        loom_with_post_pre = np.concatenate((np.zeros(int(seconds_pre * _settings['loom_hz'])), loom_with_post))
                        t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                        axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                        axs[loom_trace_row, stim_counter].set_ylim([0, 180])
                        axs[loom_trace_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                        axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                        axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                        axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                        axs[loom_trace_row, stim_counter].set_yticks([])
            else:
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Yaw_Outer_dFF_L_stim.append(right_dFF_all)
                        Yaw_Inner_dFF_L_stim.append(left_dFF_all)
                        Yaw_Outer_wing_L_stim.append(Rwing_all)
                        Yaw_Inner_wing_L_stim.append(Lwing_all)
                        Yaw_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Yaw_Outer_dFF_R_stim.append(left_dFF_all)
                        Yaw_Inner_dFF_R_stim.append(right_dFF_all)
                        Yaw_Outer_wing_R_stim.append(Lwing_all)
                        Yaw_Inner_wing_R_stim.append(Rwing_all)
                        Yaw_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Roll_Outer_dFF_L_stim.append(right_dFF_all)
                        Roll_Inner_dFF_L_stim.append(left_dFF_all)
                        Roll_Outer_wing_L_stim.append(Rwing_all)
                        Roll_Inner_wing_L_stim.append(Lwing_all)
                        Roll_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Roll_Outer_dFF_R_stim.append(left_dFF_all)
                        Roll_Inner_dFF_R_stim.append(right_dFF_all)
                        Roll_Outer_wing_R_stim.append(Lwing_all)
                        Roll_Inner_wing_R_stim.append(Rwing_all)
                        Roll_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if patterns_to_plot[stim_keys].split(' ')[1] == 'left':
                        Loom_Outer_dFF_L_stim.append(right_dFF_all)
                        Loom_Inner_dFF_L_stim.append(left_dFF_all)
                        Loom_Outer_wing_L_stim.append(Rwing_all)
                        Loom_Inner_wing_L_stim.append(Lwing_all)
                        Loom_LpRwing_L_stim.append(LpRwing_all)
                    else:
                        Loom_Outer_dFF_R_stim.append(left_dFF_all)
                        Loom_Inner_dFF_R_stim.append(right_dFF_all)
                        Loom_Outer_wing_R_stim.append(Lwing_all)
                        Loom_Inner_wing_R_stim.append(Rwing_all)
                        Loom_LpRwing_R_stim.append(LpRwing_all)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Yaw':
                    if len(Yaw_Outer_dFF_L_stim) == 0 or len(Yaw_Outer_dFF_R_stim) == 0:
                        continue
                    else:
                        Yaw_Outer_dFF_all = [np.vstack(Yaw_Outer_dFF_L_stim), np.vstack(Yaw_Outer_dFF_R_stim)]
                        Yaw_Outer_dFF_all = np.vstack(Yaw_Outer_dFF_all)
                        Yaw_Inner_dFF_all = [np.vstack(Yaw_Inner_dFF_L_stim), np.vstack(Yaw_Inner_dFF_R_stim)]
                        Yaw_Inner_dFF_all = np.vstack(Yaw_Inner_dFF_all)
                        Yaw_Outer_wing_all = [np.vstack(Yaw_Outer_wing_L_stim), np.vstack(Yaw_Outer_wing_R_stim)]
                        Yaw_Outer_wing_all = np.vstack(Yaw_Outer_wing_all)
                        Yaw_Inner_wing_all = [np.vstack(Yaw_Inner_wing_L_stim), np.vstack(Yaw_Inner_wing_R_stim)]
                        Yaw_Inner_wing_all = np.vstack(Yaw_Inner_wing_all)
                        Yaw_LpRwing_all = [np.vstack(Yaw_LpRwing_L_stim), np.vstack(Yaw_LpRwing_R_stim)]
                        Yaw_LpRwing_all = np.vstack(Yaw_LpRwing_all)
                        Outer_dFF_mean = np.nanmean(Yaw_Outer_dFF_all, axis=0)
                        Inner_dFF_mean = np.nanmean(Yaw_Inner_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Yaw_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Yaw_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Yaw_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ': ' + str(len(LpRwing_all)))
                        if plot_bs_CI == True:
                            CI_dFF_Outer = create_CI_timetrace(np.vstack(Yaw_Outer_dFF_all), iterations=1000, CI=95.0)
                            CI_dFF_Inner = create_CI_timetrace(np.vstack(Yaw_Inner_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Yaw_LpRwing_all), iterations=1000, CI=95.0)
                            if include_LmR:
                                CI_lmr = create_CI_timetrace(np.vstack(Yaw_Inner_wing_all) - np.vstack(Yaw_Outer_wing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Outer[:, 0], CI_dFF_Outer[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Inner[:, 0], CI_dFF_Inner[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            if include_LmR:
                                axs[LmRwing_row, stim_counter].fill_between(t_int, CI_lmr[:, 0], CI_lmr[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, Outer_dFF_mean, color=_settings['Owing_hexcode'], linewidth=line_width)
                        axs[dff_row, stim_counter].plot(t_int, Inner_dFF_mean, color=_settings['Iwing_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        if include_LmR:
                            axs[LmRwing_row, stim_counter].plot(t_int, Iwing_mean - Owing_mean, color=_settings['lrwing_hexcode'], label='L-R wing', linewidth=line_width)
                            axs[LmRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[0, stim_counter].set_title('Yaw\n ', fontsize=axis_label_fontsize)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Roll':
                    if len(Roll_Outer_dFF_L_stim) == 0 or len(Roll_Outer_dFF_R_stim) == 0:
                        continue
                    else:
                        Roll_Outer_dFF_all = [np.vstack(Roll_Outer_dFF_L_stim), np.vstack(Roll_Outer_dFF_R_stim)]
                        Roll_Outer_dFF_all = np.vstack(Roll_Outer_dFF_all)
                        Roll_Inner_dFF_all = [np.vstack(Roll_Inner_dFF_L_stim), np.vstack(Roll_Inner_dFF_R_stim)]
                        Roll_Inner_dFF_all = np.vstack(Roll_Inner_dFF_all)
                        Roll_Outer_wing_all = [np.vstack(Roll_Outer_wing_L_stim), np.vstack(Roll_Outer_wing_R_stim)]
                        Roll_Outer_wing_all = np.vstack(Roll_Outer_wing_all)
                        Roll_Inner_wing_all = [np.vstack(Roll_Inner_wing_L_stim), np.vstack(Roll_Inner_wing_R_stim)]
                        Roll_Inner_wing_all = np.vstack(Roll_Inner_wing_all)
                        Roll_LpRwing_all = [np.vstack(Roll_LpRwing_L_stim), np.vstack(Roll_LpRwing_R_stim)]
                        Roll_LpRwing_all = np.vstack(Roll_LpRwing_all)
                        Outer_dFF_mean = np.nanmean(Roll_Outer_dFF_all, axis=0)
                        Inner_dFF_mean = np.nanmean(Roll_Inner_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Roll_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Roll_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Roll_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ': ' + str(len(LpRwing_all)))
                        if plot_bs_CI == True:
                            CI_dFF_Outer = create_CI_timetrace(np.vstack(Roll_Outer_dFF_all), iterations=1000, CI=95.0)
                            CI_dFF_Inner = create_CI_timetrace(np.vstack(Roll_Inner_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Roll_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Roll_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Roll_LpRwing_all), iterations=1000, CI=95.0)
                            if include_LmR:
                                CI_lmr = create_CI_timetrace(np.vstack(Roll_Inner_wing_all) - np.vstack(Roll_Outer_wing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Outer[:, 0], CI_dFF_Outer[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Inner[:, 0], CI_dFF_Inner[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                            if include_LmR:
                                axs[LmRwing_row, stim_counter].fill_between(t_int, CI_lmr[:, 0], CI_lmr[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, Outer_dFF_mean, color=_settings['Owing_hexcode'], linewidth=line_width)
                        axs[dff_row, stim_counter].plot(t_int, Inner_dFF_mean, color=_settings['Iwing_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        if include_LmR:
                            axs[LmRwing_row, stim_counter].plot(t_int, Iwing_mean - Owing_mean, color=_settings['lrwing_hexcode'], label='L-R wing', linewidth=line_width)
                            axs[LmRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[0, stim_counter].set_title('Roll\n ', fontsize=axis_label_fontsize)
                if patterns_to_plot[stim_keys].split(' ')[0] == 'Loom':
                    if include_loom_trace == True:
                        stimulus_seconds = _settings['loom_duration']
                    if len(Loom_Outer_dFF_L_stim) == 0 or len(Loom_Outer_dFF_R_stim) == 0:
                        continue
                    else:
                        Loom_Outer_dFF_all = [np.vstack(Loom_Outer_dFF_L_stim), np.vstack(Loom_Outer_dFF_R_stim)]
                        Loom_Outer_dFF_all = np.vstack(Loom_Outer_dFF_all)
                        Loom_Inner_dFF_all = [np.vstack(Loom_Inner_dFF_L_stim), np.vstack(Loom_Inner_dFF_R_stim)]
                        Loom_Inner_dFF_all = np.vstack(Loom_Inner_dFF_all)
                        Loom_Outer_wing_all = [np.vstack(Loom_Outer_wing_L_stim), np.vstack(Loom_Outer_wing_R_stim)]
                        Loom_Outer_wing_all = np.vstack(Loom_Outer_wing_all)
                        Loom_Inner_wing_all = [np.vstack(Loom_Inner_wing_L_stim), np.vstack(Loom_Inner_wing_R_stim)]
                        Loom_Inner_wing_all = np.vstack(Loom_Inner_wing_all)
                        Loom_LpRwing_all = [np.vstack(Loom_LpRwing_L_stim), np.vstack(Loom_LpRwing_R_stim)]
                        Loom_LpRwing_all = np.vstack(Loom_LpRwing_all)
                        Outer_dFF_mean = np.nanmean(Loom_Outer_dFF_all, axis=0)
                        Inner_dFF_mean = np.nanmean(Loom_Inner_dFF_all, axis=0)
                        Iwing_mean = np.nanmean(Loom_Inner_wing_all, axis=0)
                        Owing_mean = np.nanmean(Loom_Outer_wing_all, axis=0)
                        LpRwing_mean = np.nanmean(Loom_LpRwing_all, axis=0)
                        print(patterns_to_plot[stim_keys].split(' ')[0] + ': ' + str(len(LpRwing_all)))
                        if plot_bs_CI == True:
                            CI_dFF_Outer = create_CI_timetrace(np.vstack(Loom_Outer_dFF_all), iterations=1000, CI=95.0)
                            CI_dFF_Inner = create_CI_timetrace(np.vstack(Loom_Inner_dFF_all), iterations=1000, CI=95.0)
                            CI_Iwing = create_CI_timetrace(np.vstack(Loom_Inner_wing_all), iterations=1000, CI=95.0)
                            CI_Owing = create_CI_timetrace(np.vstack(Loom_Outer_wing_all), iterations=1000, CI=95.0)
                            CI_lrwing = create_CI_timetrace(np.vstack(Loom_LpRwing_all), iterations=1000, CI=95.0)
                            if include_LmR:
                                CI_lmr = create_CI_timetrace(np.vstack(Loom_Inner_wing_all) - np.vstack(Loom_Outer_wing_all), iterations=1000, CI=95.0)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Iwing[:, 0], CI_Iwing[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[wing_row, stim_counter].fill_between(t_int, CI_Owing[:, 0], CI_Owing[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[LRwing_row, stim_counter].fill_between(t_int, CI_lrwing[:, 0], CI_lrwing[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Outer[:, 0], CI_dFF_Outer[:, 1], color=_settings['Owing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            axs[dff_row, stim_counter].fill_between(t_int, CI_dFF_Inner[:, 0], CI_dFF_Inner[:, 1], color=_settings['Iwing_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                            if include_LmR:
                                axs[LmRwing_row, stim_counter].fill_between(t_int, CI_lmr[:, 0], CI_lmr[:, 1], color=_settings['lrwing_hexcode'], alpha=alpha_CI, linewidth=0.0001)
                        axs[LRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[wing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        axs[dff_row, stim_counter].plot(t_int, Outer_dFF_mean, color=_settings['Owing_hexcode'], linewidth=line_width)
                        axs[dff_row, stim_counter].plot(t_int, Inner_dFF_mean, color=_settings['Iwing_hexcode'], linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Owing_mean, color=_settings['Owing_hexcode'], label='Outer wing', linewidth=line_width)
                        axs[wing_row, stim_counter].plot(t_int, Iwing_mean, color=_settings['Iwing_hexcode'], label='Inner wing', linewidth=line_width)
                        axs[LRwing_row, stim_counter].plot(t_int, LpRwing_mean, color=_settings['lrwing_hexcode'], label='L+R wing', linewidth=line_width)
                        if include_LmR:
                            axs[LmRwing_row, stim_counter].plot(t_int, Iwing_mean - Owing_mean, color=_settings['lrwing_hexcode'], label='L-R wing', linewidth=line_width)
                            axs[LmRwing_row, stim_counter].fill_between([0, stimulus_seconds], -30, 130, color=_settings['visual_stim_hexcode'], alpha=alpha_patch, linewidth=0.0001)
                        if include_loom_trace == True:
                            loom_radians = np.loadtxt(loom_path(), delimiter=',')
                            loom_deg = loom_radians * (180 / np.pi)
                            loom_deg[loom_deg < 0] = 0
                            loom_with_post = np.concatenate((loom_deg, np.zeros(seconds_post * _settings['loom_hz'])))
                            loom_with_post_pre = np.concatenate((np.zeros(int(seconds_pre * _settings['loom_hz'])), loom_with_post))
                            t_loom = np.linspace(start=0 - seconds_pre, stop=_settings['loom_duration'] + seconds_post, num=int(len(loom_radians) + seconds_post * _settings['loom_hz'] + seconds_pre * _settings['loom_hz']))
                            axs[loom_trace_row, stim_counter].plot(t_loom, loom_with_post_pre, color='k', linewidth=line_width, alpha=0.5)
                            axs[loom_trace_row, stim_counter].set_ylim([0, 180])
                            axs[loom_trace_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                            axs[loom_trace_row, stim_counter].spines.left.set_bounds((0, 90))
                            axs[loom_trace_row, stim_counter].set_ylabel('90 deg', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, va='bottom')
                            axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                            axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                            axs[loom_trace_row, stim_counter].set_yticks([])
                        axs[0, stim_counter].set_title('Lateral \nloom', fontsize=axis_label_fontsize)
                else:
                    axs[loom_trace_row, stim_counter].spines['top'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['right'].set_visible(False)
                    axs[loom_trace_row, stim_counter].spines['left'].set_visible(False)
                    axs[loom_trace_row, stim_counter].set_yticks([])
            axs[wing_row, stim_counter].set_ylim([-15, 15])
            axs[LRwing_row, stim_counter].set_ylim([-20, 20])
            axs[dff_row, stim_counter].set_ylim([-2, 8])
            axs[wing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
            axs[LRwing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
            axs[dff_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
            if include_LmR:
                axs[LmRwing_row, stim_counter].set_xlim([-1.5, stimulus_seconds + seconds_post])
                axs[LmRwing_row, stim_counter].set_ylim([-25, 15])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_bounds((0, round(stimulus_seconds)))
            if stim_counter == 0:
                axs[dff_row, stim_counter].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[wing_row, stim_counter].set_ylabel('wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[LRwing_row, stim_counter].set_ylabel('L+R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                axs[dff_row, stim_counter].set_yticks([0, 2.5, 5])
                axs[wing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[LRwing_row, stim_counter].set_yticks([-10, 0, 10])
                axs[dff_row, stim_counter].set_yticklabels(axs[dff_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].set_yticklabels(axs[wing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[LRwing_row, stim_counter].set_yticklabels(axs[LRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                if include_LmR:
                    axs[LmRwing_row, stim_counter].set_ylabel('L-R wing \n amplitude \n (deg)', rotation=90, fontsize=axis_label_fontsize, labelpad=label_pad, ha='center')
                    axs[LmRwing_row, stim_counter].set_yticks([-20, -10, 0, 10])
                    axs[LmRwing_row, stim_counter].set_yticklabels(axs[LmRwing_row, stim_counter].get_yticks(), fontsize=axis_ticklabel_fontsize)
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(True)
                if include_LmR:
                    axs[LmRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[LmRwing_row, stim_counter].spines['bottom'].set_visible(False)
                    axs[LmRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[rows_stim - 1, stim_counter].spines['bottom'].set_visible(True)
                axs[rows_stim - 1, stim_counter].set_xlabel('time (s)', fontsize=axis_label_fontsize, va='center')
                axs[rows_stim - 1, stim_counter].set_xticks([0, np.round(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                for j in range(rows_stim - 1):
                    axs[j, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[LRwing_row, stim_counter].spines.left.set_bounds((-10, 10))
                axs[dff_row, stim_counter].spines.left.set_bounds((0, 5))
                if include_LmR:
                    axs[LmRwing_row, stim_counter].spines.left.set_bounds((-20, 10))
            else:
                axs[dff_row, stim_counter].set_yticks([])
                axs[wing_row, stim_counter].set_yticks([])
                axs[LRwing_row, stim_counter].set_yticks([])
                if include_LmR:
                    axs[LmRwing_row, stim_counter].set_yticks([])
                for j in range(rows_stim - 1):
                    axs[j, stim_counter].set_xticks([])
                axs[wing_row, stim_counter].spines['top'].set_visible(False)
                axs[wing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[wing_row, stim_counter].spines['right'].set_visible(False)
                axs[wing_row, stim_counter].spines['left'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['top'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['right'].set_visible(False)
                axs[LRwing_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['top'].set_visible(False)
                axs[dff_row, stim_counter].spines['right'].set_visible(False)
                axs[dff_row, stim_counter].spines['left'].set_visible(False)
                axs[dff_row, stim_counter].spines['bottom'].set_visible(False)
                if include_LmR:
                    axs[LmRwing_row, stim_counter].spines['top'].set_visible(False)
                    axs[LmRwing_row, stim_counter].spines['right'].set_visible(False)
                    axs[LmRwing_row, stim_counter].spines['left'].set_visible(False)
                    axs[LmRwing_row, stim_counter].spines['bottom'].set_visible(False)
                axs[rows_stim - 1, stim_counter].set_xticks([0, round(stimulus_seconds)])
                axs[rows_stim - 1, stim_counter].set_xticklabels(axs[rows_stim - 1, stim_counter].get_xticks(), fontsize=axis_ticklabel_fontsize)
                axs[rows_stim - 1, stim_counter].spines['bottom'].set_visible(True)
            for j in range(rows_stim):
                axs[j, stim_counter].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
                axs[j, stim_counter].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[wing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[LRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[dff_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            if include_LmR:
                axs[LmRwing_row, stim_counter].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[rows_stim - 1, stim_counter].spines.bottom.set_linewidth(_settings['line_width_spines'])
            stim_counter += 1
    fig.align_ylabels(axs[:])
    fig.set_size_inches(column_width, row_height)
    fig.tight_layout()
    if _settings['layout_mode'] != 'CurrBio':
        fig.subplots_adjust(wspace=0.02, hspace=0.22)
    else:
        fig.subplots_adjust(left=0.082, bottom=0.096, right=0.983, top=0.915, wspace=0.0, hspace=0.252)
    return dict(figure=fig, results={})

def plot_avg_trace_per_flight_starts_per_trial_combined_flies_single_roi(datasets, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, bout_time=2, plot_bs_CI=True, line_width=1, axis_ticklabel_fontsize=6, axis_label_fontsize=6):
    """
    generate avg trace plots for all data combined
    single roi dFF
    organized in fly>trial>trialnum>data
    take mean of fly, mean of means 
    here bout time and seconds post are combined into 1 period
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    rows_stim = 1
    columns_stim = 1
    fig, axs = plt.subplots(rows_stim, columns_stim, squeeze=False)
    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * bout_time)
    frames_one_s = int(imRateHz_mean * 1)
    for stim_id in range(1):
        column = 0
        row_dff = 0
        dFF_all = []
        Lwing_all = []
        Rwing_all = []
        LpRwing_all = []
        dFF_fly_means_over_bout_all = []
        dFF_fly_means_over_last_s_all = []
        for fly_id in all_dFF_df['fly_id'].unique():
            stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
            dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
            dFF_fly = []
            Lwing_fly = []
            Rwing_fly = []
            LpRwing_fly = []
            for trial in dFF_df_fly['trial_num'].unique():
                stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
                dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
                flying_threshold = flying_threshold_default
                flight_starts, flight_stops = get_times_flight_stop_start(stim_df, flying_threshold=flying_threshold, bout_time=bout_time + seconds_post, frames_consecutive=5, seconds_pre=seconds_pre)
                if stim_id == 0:
                    change_times = flight_starts
                for start in change_times:
                    start_t = start - seconds_pre
                    end_t = start + bout_time + seconds_post
                    pre_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF'].values
                    pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                    L_pre_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                    R_pre_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                    pre_wing_times = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                    stim_dFF = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end_t)]['dFF'].values
                    stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                    L_stim_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                    R_stim_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                    stim_wing_times = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                    pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                    stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                    if len(pre_dFF_times) != 0 and len(stim_dFF_times) != 0:
                        t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                        t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                        t_wing_train_int = np.concatenate((pre_wing_times, stim_wing_times)) - start
                        dFF_train_int = np.concatenate((pre_dFF, stim_dFF))
                        dFF_int = interpolate_trace(t_dFF_train_int, dFF_train_int, t_int)
                        baseline_dFF_int = np.nanmean(dFF_int[:frames_pre])
                        dFF_fly.append(dFF_int - baseline_dFF_int)
                        Lwing_train_int = np.concatenate((L_pre_wing, L_stim_wing))
                        Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                        Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                        Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                        Rwing_train_int = np.concatenate((R_pre_wing, R_stim_wing))
                        Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                        Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                        Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                        LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing))
                        LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                        LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                        LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
            dFF_fly_mean = np.nanmean(dFF_fly, axis=0)
            Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
            Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
            LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
            dFF_fly_mean_over_bout = np.nanmean(np.array(dFF_fly)[:, frames_pre:frames_pre + frames_stim + frames_post])
            dFF_fly_means_over_bout_all.append(dFF_fly_mean_over_bout)
            dFF_fly_mean_over_last_s = np.nanmean(np.array(dFF_fly)[:, frames_pre + frames_stim + frames_post - frames_one_s:frames_pre + frames_stim + frames_post])
            dFF_fly_means_over_last_s_all.append(dFF_fly_mean_over_last_s)
            if hasattr(dFF_fly_mean, '__len__'):
                dFF_all.append(dFF_fly_mean)
                Lwing_all.append(Lwing_fly_mean)
                Rwing_all.append(Rwing_fly_mean)
                LpRwing_all.append(LpRwing_fly_mean)
        if len(dFF_all) != 0:
            dFF_mean = np.nanmean(dFF_all, axis=0)
            Lwing_mean = np.nanmean(Lwing_all, axis=0)
            Rwing_mean = np.nanmean(Rwing_all, axis=0)
            LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
            print(f'flight starts N={len(dFF_all)}')
            if plot_bs_CI == True:
                CI_dFF = create_CI_timetrace(np.vstack(dFF_all), iterations=1000, CI=95.0)
                CI_Lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
                CI_Rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
                CI_LpRwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
                axs[row_dff, column].fill_between(t_int, CI_dFF[:, 0], CI_dFF[:, 1], color=_settings['bs_CI_hexcode'], alpha=0.5)
            axs[row_dff, column].plot(t_int, dFF_mean, color='black', alpha=1, linewidth=line_width)
            axs[row_dff, column].set_title('flight onset', fontsize=axis_label_fontsize)
            axs[row_dff, column].spines['top'].set_visible(False)
            axs[row_dff, column].set_ylim([-0.25, 0.5])
            axs[row_dff, column].set_yticks([0, 0.5])
            axs[row_dff, column].set_xlim([0 - seconds_pre, bout_time + seconds_post])
            axs[row_dff, column].set_xticks([0, bout_time + seconds_post])
            axs[row_dff, column].spines.bottom.set_bounds((0, bout_time + seconds_post))
            axs[row_dff, column].spines['right'].set_visible(False)
            axs[row_dff, column].spines.left.set_bounds((-0.5, 3))
            axs[row_dff, column].xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[row_dff, column].yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
            axs[row_dff, column].spines.left.set_linewidth(_settings['line_width_spines'])
            axs[row_dff, column].spines.bottom.set_linewidth(_settings['line_width_spines'])
            axs[row_dff, column].set_xticklabels(axs[row_dff, column].get_xticks(), fontsize=axis_ticklabel_fontsize)
            axs[row_dff, column].set_yticklabels(axs[row_dff, column].get_yticks(), fontsize=axis_ticklabel_fontsize)
            axs[row_dff, column].set_xlabel('time (s)', fontsize=axis_label_fontsize)
            axs[row_dff, column].set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center')
    fig.set_size_inches(_settings['column_width_2'] * 0.3, _settings['column_width_2'] * 0.15)
    fig.subplots_adjust(left=0.373, bottom=0.173, right=0.843, top=0.964)
    return dict(figure=fig, results={})

def plot_avg_trace_per_flight_starts_combined_flies_descending_branches(datasets, line='1070', flying_threshold_default=0.14, seconds_pre=1, seconds_post=1, airpuff_threshold=2.0, bout_time=2, plot_bs_CI=True, line_width=1, axis_ticklabel_fontsize=6, axis_label_fontsize=6):
    """
    generate avg trace plots for all data combined
    2 roi dFF
    original stimuli
    organized in fly>trial>trialnum>data
    take mean of fly, mean of means 
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_general_df_dir'], line)
    rows_stim = 1
    columns_stim = 1
    fig, axs = plt.subplots(rows_stim, columns_stim)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    frames_pre = int(imRateHz_mean * seconds_pre)
    frames_post = int(imRateHz_mean * seconds_post)
    frames_stim = int(imRateHz_mean * bout_time)
    dFF_all_L = []
    dFF_all_R = []
    Lwing_all = []
    Rwing_all = []
    LpRwing_all = []
    for fly_id in all_dFF_df['fly_id'].unique():
        stim_df_fly = all_stim_df[all_stim_df['fly_id'] == fly_id]
        dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
        dFF_fly_L = []
        dFF_fly_R = []
        Lwing_fly = []
        Rwing_fly = []
        LpRwing_fly = []
        for trial in dFF_df_fly['trial_num'].unique():
            stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
            dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
            flying_threshold = flying_threshold_default
            flight_starts, flight_stops = get_times_flight_stop_start(stim_df, flying_threshold=flying_threshold, bout_time=bout_time + seconds_post, frames_consecutive=5, seconds_pre=seconds_pre)
            change_times = flight_starts
            for start in change_times:
                start_t = start - seconds_pre
                end_t = start + bout_time + seconds_post
                pre_dFF_L = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_left'].values
                pre_dFF_R = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['dFF_right'].values
                pre_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start_t) & (dFF_df['frameTimes'] < start)]['frameTimes'].values
                L_pre_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['LW_filt'].values
                R_pre_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['RW_filt'].values
                pre_wing_times = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['timeVec'].values
                stim_dFF_L = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end_t)]['dFF_left'].values
                stim_dFF_R = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end_t)]['dFF_right'].values
                stim_dFF_times = dFF_df.loc[(dFF_df['frameTimes'] >= start) & (dFF_df['frameTimes'] < end_t)]['frameTimes'].values
                L_stim_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['LW_filt'].values
                R_stim_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['RW_filt'].values
                stim_wing_times = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['timeVec'].values
                pre_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start_t) & (stim_df['timeVec'] < start)]['L+R_filt'].values
                stim_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= start) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                post_left_plus_right_wing = stim_df.loc[(stim_df['timeVec'] >= end_t) & (stim_df['timeVec'] < end_t)]['L+R_filt'].values
                if len(pre_dFF_times) != 0 and len(stim_dFF_times) != 0:
                    t_int = np.linspace(start_t - start, end_t - start, frames_pre + frames_stim + frames_post)
                    t_dFF_train_int = np.concatenate((pre_dFF_times, stim_dFF_times)) - start
                    t_wing_train_int = np.concatenate((pre_wing_times, stim_wing_times)) - start
                    dFF_train_int_L = np.concatenate((pre_dFF_L, stim_dFF_L))
                    dFF_train_int_R = np.concatenate((pre_dFF_R, stim_dFF_R))
                    dFF_int_L = interpolate_trace(t_dFF_train_int, dFF_train_int_L, t_int)
                    baseline_dFF_int_L = np.nanmean(dFF_int_L[:frames_pre])
                    dFF_fly_L.append(dFF_int_L - baseline_dFF_int_L)
                    dFF_int_R = interpolate_trace(t_dFF_train_int, dFF_train_int_R, t_int)
                    baseline_dFF_int_R = np.nanmean(dFF_int_R[:frames_pre])
                    dFF_fly_R.append(dFF_int_R - baseline_dFF_int_R)
                    Lwing_train_int = np.concatenate((L_pre_wing, L_stim_wing))
                    Lwing_int = interpolate_trace(t_wing_train_int, Lwing_train_int, t_int)
                    Lbaseline_wing_int = np.nanmean(Lwing_int[:frames_pre])
                    Lwing_fly.append(Lwing_int - Lbaseline_wing_int)
                    Rwing_train_int = np.concatenate((R_pre_wing, R_stim_wing))
                    Rwing_int = interpolate_trace(t_wing_train_int, Rwing_train_int, t_int)
                    Rbaseline_wing_int = np.nanmean(Rwing_int[:frames_pre])
                    Rwing_fly.append(Rwing_int - Rbaseline_wing_int)
                    LpRwing_train_int = np.concatenate((pre_left_plus_right_wing, stim_left_plus_right_wing, post_left_plus_right_wing))
                    LpRwing_int = interpolate_trace(t_wing_train_int, LpRwing_train_int, t_int)
                    LpRbaseline_wing_int = np.nanmean(LpRwing_int[:frames_pre])
                    LpRwing_fly.append(LpRwing_int - LpRbaseline_wing_int)
        dFF_fly_mean_L = np.nanmean(dFF_fly_L, axis=0)
        dFF_fly_mean_R = np.nanmean(dFF_fly_R, axis=0)
        Lwing_fly_mean = np.nanmean(Lwing_fly, axis=0)
        Rwing_fly_mean = np.nanmean(Rwing_fly, axis=0)
        LpRwing_fly_mean = np.nanmean(LpRwing_fly, axis=0)
        if hasattr(dFF_fly_mean_L, '__len__'):
            dFF_all_L.append(dFF_fly_mean_L)
            dFF_all_R.append(dFF_fly_mean_R)
            Lwing_all.append(Lwing_fly_mean)
            Rwing_all.append(Rwing_fly_mean)
            LpRwing_all.append(LpRwing_fly_mean)
    if len(dFF_all_L) != 0:
        dFF_mean_L = np.nanmean(dFF_all_L, axis=0)
        dFF_mean_R = np.nanmean(dFF_all_R, axis=0)
        Lwing_mean = np.nanmean(Lwing_all, axis=0)
        Rwing_mean = np.nanmean(Rwing_all, axis=0)
        LpRwing_mean = np.nanmean(LpRwing_all, axis=0)
        print(f'N={np.shape(dFF_all_L)[0]}')
        if plot_bs_CI == True:
            CI_dFF_L = create_CI_timetrace(np.vstack(dFF_all_L), iterations=1000, CI=95.0)
            CI_dFF_R = create_CI_timetrace(np.vstack(dFF_all_R), iterations=1000, CI=95.0)
            CI_Lwing = create_CI_timetrace(np.vstack(Lwing_all), iterations=1000, CI=95.0)
            CI_Rwing = create_CI_timetrace(np.vstack(Rwing_all), iterations=1000, CI=95.0)
            CI_LpRwing = create_CI_timetrace(np.vstack(LpRwing_all), iterations=1000, CI=95.0)
            axs.fill_between(t_int, CI_dFF_L[:, 0], CI_dFF_L[:, 1], color=_settings['left_dFF_hexcode'], alpha=_settings['alpha_CI_all'], edgecolor='none')
            axs.fill_between(t_int, CI_dFF_R[:, 0], CI_dFF_R[:, 1], color=_settings['right_dFF_hexcode'], alpha=_settings['alpha_CI_all'], edgecolor='none')
        axs.plot(t_int, dFF_mean_L, color=_settings['left_dFF_hexcode'], alpha=1, linewidth=line_width)
        axs.plot(t_int, dFF_mean_R, color=_settings['right_dFF_hexcode'], alpha=1, linewidth=line_width)
        axs.spines['top'].set_visible(False)
        axs.set_ylim([-0.5, 3])
        axs.set_yticks([-0.5, 0, 1, 2, 3])
        axs.set_xlim([0 - seconds_pre - 0.25, bout_time + seconds_post])
        axs.set_xticks([-seconds_pre, 0, bout_time + seconds_post])
        axs.spines.bottom.set_bounds((-seconds_pre, bout_time + seconds_post))
        axs.spines['right'].set_visible(False)
        axs.spines.left.set_bounds((-0.5, 3))
        axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
        axs.spines.left.set_linewidth(_settings['line_width_spines'])
        axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
        axs.set_xticklabels(axs.get_xticks(), fontsize=axis_ticklabel_fontsize)
        axs.set_yticklabels(axs.get_yticks(), fontsize=axis_ticklabel_fontsize)
        axs.set_title('flight starts', fontsize=axis_label_fontsize)
        axs.set_ylabel('$\\Delta$F/F', rotation=90, fontsize=axis_label_fontsize, ha='center')
        axs.set_xlabel('time (s)', rotation=0, fontsize=axis_label_fontsize, va='center')
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] * (1 / 6), _settings['column_width_2'] / 4)
        fig.subplots_adjust(left=0.364, bottom=0.167, right=0.93, top=0.865)
    else:
        fig.set_size_inches(_settings['column_width_3'] * (1 / 3), _settings['column_width_3'] / 4)
    return dict(figure=fig, results={})

def plot_corr_descending_left_right_dff_all_trials(datasets, line='1070', flying_threshold_default=0.14, regression='ODR', plot_z_score=False, z_score_per='trial', line_width=1, point_size=6, axis_ticklabel_fontsize=6, axis_label_fontsize=6, plot_trendline=False):
    """
    plot correlation of left and right dff during entire trial 
    can plot all points or just means per fly etc 
    regression what type of regression to run (ODR, linear)
    """
    _settings = config.load('two_photon')['settings']
    all_dFF_df, all_stim_df = load_combined(datasets['data_df_dir'], line)
    fig, axs = plt.subplots(1, 1)
    imRateHz_mean = np.median(all_dFF_df['imRateHz'].values)
    dff_left_all_points = []
    dff_right_all_points = []
    dff_left_all_z = []
    dff_right_all_z = []
    slopes_per_fly = []
    offset_per_fly = []
    dff_left_all_flies_z = []
    dff_right_all_flies_z = []
    for fly_id in all_dFF_df['fly_id'].unique():
        dff_left_fly_all = []
        dff_right_fly_all = []
        dff_left_all_trials_fly_z = []
        dff_right_all_trials_fly_z = []
        dFF_df_fly = all_dFF_df[all_dFF_df['fly_id'] == fly_id]
        for trial in dFF_df_fly['trial_num'].unique():
            stim_df = all_stim_df[(all_stim_df['fly_id'] == fly_id) & (all_stim_df['trial_num'] == trial)]
            dFF_df = all_dFF_df[(all_dFF_df['fly_id'] == fly_id) & (all_dFF_df['trial_num'] == trial)]
            flying_threshold = flying_threshold_default
            dff_left = dFF_df['dFF_left'].values
            dff_right = dFF_df['dFF_right'].values
            flying = stim_df['aux_kinefly_filt'] >= flying_threshold
            dff_left_fly_all.extend(dff_left[flying])
            dff_right_fly_all.extend(dff_right[flying])
            dff_left_trial_z = z_score(dff_left[flying])
            dff_right_trial_z = z_score(dff_right[flying])
            dff_left_all_trials_fly_z.extend(dff_left_trial_z)
            dff_right_all_trials_fly_z.extend(dff_right_trial_z)
            dff_left_all_flies_z.extend(dff_left_trial_z)
            dff_right_all_flies_z.extend(dff_right_trial_z)
        dff_left_all_points.extend(dff_left_fly_all)
        dff_right_all_points.extend(dff_right_fly_all)
        dff_left_fly_z = z_score(dff_left_fly_all)
        dff_right_fly_z = z_score(dff_right_fly_all)
        dff_left_all_z.extend(dff_left_fly_z)
        dff_right_all_z.extend(dff_right_fly_z)
        if z_score_per == 'fly':
            x_fly = dff_right_fly_z
            y_fly = dff_left_fly_z
        elif z_score_per == 'trial':
            x_fly = dff_right_all_trials_fly_z
            y_fly = dff_left_all_trials_fly_z
        else:
            raise ValueError('no valid z score per')
        if regression == 'ODR':
            x_arr = np.array(x_fly)
            y_arr = np.array(y_fly)
            x_new = x_arr[~np.isnan(y_arr)]
            y_new = y_arr[~np.isnan(y_arr)]
            sx = np.nanstd(x_new)
            sy = np.nanstd(y_new)
            output = orthogonal_distance_regression(x_new, y_new, sx, sy)
            slopes_per_fly.append(output.beta[0])
            offset_per_fly.append(output.beta[1])
    if plot_z_score == False:
        axs.scatter(dff_right_all_points, dff_left_all_points, color='black', alpha=0.15, edgecolors='None', s=point_size)
        z_score_txt = ''
        y = dff_left_all_points
        x = dff_right_all_points
    else:
        if z_score_per == 'fly':
            x = dff_right_all_z
            y = dff_left_all_z
        elif z_score_per == 'trial':
            x = dff_right_all_flies_z
            y = dff_left_all_flies_z
        else:
            raise ValueError('not valid z score per')
        axs.scatter(x, y, color='black', alpha=0.15, edgecolors='None', s=point_size)
        z_score_txt = '_z_scored'
    x_arr = np.array(x)
    y_arr = np.array(y)
    x_new = x_arr[~np.isnan(y_arr)]
    y_new = y_arr[~np.isnan(y_arr)]
    if regression == 'linear':
        slope, intercept, r, p, std_err = stats.linregress(x_new, y_new)
        axs.plot(x_new, slope * x_new + intercept, color='red', label=f'r={np.round(r, 3)}', linewidth=line_width)
        print('slope: ' + str(slope))
    if regression == 'ODR':
        sx = np.nanstd(x_new)
        sy = np.nanstd(y_new)
        output = orthogonal_distance_regression(x_new, y_new, sx, sy)
        mask = (x_new >= -3.5) & (x_new <= 3.5)
        value_arr = output.beta[0] * np.array(x_new) + output.beta[1]
        filtered_time = x_new[mask]
        filtered_values = value_arr[mask]
        print('beta: ' + str(output.beta[0]))
        slopes_per_fly_mean = np.mean(slopes_per_fly)
        slopes_per_fly_CI = create_CI(slopes_per_fly)
        print(f'Mean m: {slopes_per_fly_mean} CI: {slopes_per_fly_CI}')
        offset_mean = np.mean(offset_per_fly)
        print(f'N flies={len(slopes_per_fly)}')
        if plot_trendline == True:
            mask = (x_new >= -3.5) & (x_new <= 3.5)
            value_arr = slopes_per_fly_mean * np.array(x_new) + offset_mean
            filtered_time = x_new[mask]
            filtered_values = value_arr[mask]
            axs.plot(filtered_time, filtered_values, color='red', linewidth=line_width)
    if plot_z_score == True:
        axs.set_xlabel('right ' + '$\\Delta$F/F z score', fontsize=axis_label_fontsize)
        axs.set_ylabel('left ' + '$\\Delta$F/F z score', fontsize=axis_label_fontsize)
    else:
        axs.set_xlabel('right ' + '$\\Delta$F/F', fontsize=axis_label_fontsize)
        axs.set_ylabel('left ' + '$\\Delta$F/F', fontsize=axis_label_fontsize)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    if plot_z_score:
        axs.set_xlim(-4.5, 4.5)
        axs.set_ylim(-4.5, 4.5)
        axs.spines.bottom.set_bounds((-4, 4))
        axs.spines.left.set_bounds((-4, 4))
        axs.set_xticks([-4, 0, 4])
        axs.set_yticks([-4, 0, 4])
    else:
        axs.set_xlim(-5, 10)
        axs.set_ylim(-5, 10)
    axs.xaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    axs.yaxis.set_tick_params(which='major', direction='in', width=_settings['tick_width'], length=_settings['tick_length'])
    axs.spines.left.set_linewidth(_settings['line_width_spines'])
    axs.spines.bottom.set_linewidth(_settings['line_width_spines'])
    axs.set_xticklabels(axs.get_xticks(), fontsize=axis_ticklabel_fontsize)
    axs.set_yticklabels(axs.get_yticks(), fontsize=axis_ticklabel_fontsize)
    fig.tight_layout()
    if _settings['layout_mode'] == 'CurrBio':
        fig.set_size_inches(_settings['column_width_2'] / 4, _settings['column_width_2'] / 4)
        fig.subplots_adjust(left=0.204, bottom=0.18, right=0.921, top=0.964)
    else:
        fig.set_size_inches(_settings['column_width_3'] / 4, _settings['column_width_3'] / 4)
    return dict(figure=fig, results={})

