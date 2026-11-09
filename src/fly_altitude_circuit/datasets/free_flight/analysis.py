"""Free-flight calculations and helpers
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from ... import stats


def create_CI(*args, **kwargs):
    return stats.create_CI(*args, **kwargs)


def permutation_test(*args, **kwargs):
    return stats.permutation_test(*args, **kwargs)

def mean_CI_data(df, index='z', time_key='time stamp', tmin=-100, tmax=3000, CI=95.0, iterations=1000, time='ms'):
    """Takes in a data frame and the index on which you want to calculate the means and bs CI
    if time='s' then tmin and tmax are in seconds, otherwise in ms
    """
    if time == 's':
        tmin = tmin * 1000
        tmax = tmax * 1000
    mean_vec = []
    upper_vec = []
    lower_vec = []
    select_time_df = df[df[time_key].between(tmin, tmax)]
    t_vec = select_time_df[time_key].unique()
    t_vec = sorted(t_vec)
    tpoint_lens = []
    for i in t_vec:
        curr_time_df = select_time_df[select_time_df['time stamp'] == i]
        tpoint_data = curr_time_df[index].values
        tpoint_mean = np.nanmean(tpoint_data)
        tpoint_lens.append(len(tpoint_data))
        tpoint_CI = create_CI(tpoint_data, CI=CI, iterations=iterations)
        mean_vec.append(tpoint_mean)
        upper_vec.append(tpoint_CI[1])
        lower_vec.append(tpoint_CI[0])
    n = tpoint_lens[0]
    if time == 's':
        t_vec = [x / 1000 for x in t_vec]
        return [t_vec, mean_vec, upper_vec, lower_vec, n]
    else:
        return [t_vec, mean_vec, upper_vec, lower_vec, n]
def filter_trajectory(df, ground_speed_threshold=2, z_velocity_threshold=2, s_tracking_data_post=1, s_tracking_data_pre=0.2):
    """
    remove trajectories with ground speed > x m/s, and z velocity > or < x m/s
    """
    for traj in df.obj_id_unique.unique():
        df_traj = df[df['obj_id_unique'] == traj]
        if df_traj['time stamp'].max() < s_tracking_data_post * 1000:
            df = df[df['obj_id_unique'] != traj]
        if df_traj['time stamp'].min() > s_tracking_data_pre * -1000:
            df = df[df['obj_id_unique'] != traj]
        if np.any(df_traj['ground speed'] > ground_speed_threshold):
            df = df[df['obj_id_unique'] != traj]
        if np.any(df_traj['zvel'] > z_velocity_threshold) or np.any(df_traj['zvel'] < -1 * z_velocity_threshold):
            df = df[df['obj_id_unique'] != traj]
    return df
def extract_metrics_tpoint(group, tpoint=0.5):
    timepoint = tpoint * 1000
    return pd.Series({'z_at_time_0': group.loc[group['time stamp'] == 0, 'z'].values[0] if (group['time stamp'] == 0).any() else np.nan, 'z_at_timepoint': group.loc[group['time stamp'] == timepoint, 'z'].values[0] if (group['time stamp'] == timepoint).any() else np.nan})
def extract_metrics_min(group):
    return pd.Series({'z_at_time_0': group.loc[group['time stamp'] == 0, 'z'].values[0] if (group['time stamp'] == 0).any() else np.nan, 'z_min': group['z'].min()})
def mean_change_alitude_and_CI(df, tpoint_compare=None):
    """
    retrieve the change in alitude per trajectory from time=0 to min(altitude) or at a timepoint (t=0.5)
    mean across trajectories and CI 
    tpoint_compare in seconds (this function will convert to ms)
    return dict of mean change, all datapoint changes, and CI
    """
    if tpoint_compare is not None:
        result_df = df.groupby('obj_id_unique').apply(extract_metrics_tpoint, tpoint_compare).reset_index()
        mean_change = np.mean(result_df['z_at_timepoint'].values - result_df['z_at_time_0'].values)
        changes = result_df['z_at_timepoint'].values - result_df['z_at_time_0'].values
        CI = create_CI(changes)
        print(f'mean change z (at timepoint {tpoint_compare}): {mean_change * 100} cm; CI {CI * 100}')
    else:
        result_df = df.groupby('obj_id_unique').apply(extract_metrics_min).reset_index()
        mean_change = np.mean(result_df['z_min'].values - result_df['z_at_time_0'].values)
        changes = result_df['z_min'].values - result_df['z_at_time_0'].values
        CI = create_CI(changes)
        print(f'mean change z (to min): {mean_change * 100} cm; CI {CI * 100}')
    return {'mean_change': mean_change, 'altitude_change_data': changes, 'CI': CI}

def calculate_traces(df_exp, df_control, panel):
    start = panel['start']
    end = panel['end']
    time = panel['time']
    change_in_altitude_tpoint = panel['change_in_altitude_tpoint']
    natural_flight_t_endpoint = panel['natural_flight_t_endpoint']
    df_exp = filter_trajectory(df_exp)
    df_control = filter_trajectory(df_control)
    df_exp_natural = filter_trajectory(df_exp, s_tracking_data_post=natural_flight_t_endpoint, s_tracking_data_pre=0.2)
    df_control_natural = filter_trajectory(df_control, s_tracking_data_post=natural_flight_t_endpoint, s_tracking_data_pre=0.2)
    df_exp_real = df_exp[df_exp['intensity'] == 100]
    df_exp_sham = df_exp[df_exp['intensity'] == 0]
    df_control_real = df_control[df_control['intensity'] == 100]
    df_control_sham = df_control[df_control['intensity'] == 0]
    df_exp_real_natural = df_exp_natural[df_exp_natural['intensity'] == 100]
    df_exp_sham_natural = df_exp_natural[df_exp_natural['intensity'] == 0]
    df_control_real_natural = df_control_natural[df_control_natural['intensity'] == 100]
    df_control_sham_natural = df_control_natural[df_control_natural['intensity'] == 0]
    df_exp_real_mean_CI = mean_CI_data(df_exp_real, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_exp_sham_mean_CI = mean_CI_data(df_exp_sham, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_control_real_mean_CI = mean_CI_data(df_control_real, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_control_sham_mean_CI = mean_CI_data(df_control_sham, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    print(f'DNa10 light on trajectory n: {df_exp_real_mean_CI[-1]}')
    print(f'DNa10 light off trajectory n: {df_exp_sham_mean_CI[-1]}')
    print(f'Control light on trajectory n: {df_control_real_mean_CI[-1]}')
    print(f'Control light off trajectory n: {df_control_sham_mean_CI[-1]}')
    print('Exp real')
    experimental_summary = mean_change_alitude_and_CI(df_exp_real, tpoint_compare=change_in_altitude_tpoint)
    print('All control groups combined')
    df_all_controls = pd.concat([df_control_real_natural, df_exp_sham_natural, df_control_sham_natural])
    natural_summary = mean_change_alitude_and_CI(df_all_controls, tpoint_compare=natural_flight_t_endpoint)
    cohorts = {name: {'trajectory_ids': df['obj_id_unique'].unique().tolist(),
                             'trajectory_n': int(df['obj_id_unique'].nunique()), 'rows': len(df)}
               for name, df in [('exp_real', df_exp_real), ('exp_sham', df_exp_sham),
                                ('control_real', df_control_real), ('control_sham', df_control_sham)]}
    cohorts['natural_controls'] = {'trajectory_ids': df_all_controls['obj_id_unique'].unique().tolist(), 'trajectory_n': int(df_all_controls['obj_id_unique'].nunique()), 'rows': len(df_all_controls)}
    return dict(traces={'df_exp_real_mean_CI': df_exp_real_mean_CI, 'df_exp_sham_mean_CI': df_exp_sham_mean_CI, 'df_control_real_mean_CI': df_control_real_mean_CI, 'df_control_sham_mean_CI': df_control_sham_mean_CI}, cohorts=cohorts, experimental_summary=experimental_summary, natural_summary=natural_summary)

def calculate_statistics(df_exp, df_control, panel):
    start = panel['start']
    end = panel['end']
    time = panel['time']
    change_in_altitude_tpoint = panel['change_in_altitude_tpoint']
    df_exp = filter_trajectory(df_exp)
    df_control = filter_trajectory(df_control)
    df_exp_real = df_exp[df_exp['intensity'] == 100]
    df_exp_sham = df_exp[df_exp['intensity'] == 0]
    df_control_real = df_control[df_control['intensity'] == 100]
    df_control_sham = df_control[df_control['intensity'] == 0]
    df_exp_real_mean_CI = mean_CI_data(df_exp_real, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_exp_sham_mean_CI = mean_CI_data(df_exp_sham, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_control_real_mean_CI = mean_CI_data(df_control_real, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    df_control_sham_mean_CI = mean_CI_data(df_control_sham, index='z', time_key='time stamp', tmin=start, tmax=end, CI=95.0, iterations=1000, time=time)
    print(f'DNa10 light on trajectory n: {df_exp_real_mean_CI[-1]}')
    print(f'DNa10 light off trajectory n: {df_exp_sham_mean_CI[-1]}')
    print(f'Control light on trajectory n: {df_control_real_mean_CI[-1]}')
    print(f'Control light off trajectory n: {df_control_sham_mean_CI[-1]}')
    args = []
    name_list = []
    print('Exp sham')
    exp_sham_dict = mean_change_alitude_and_CI(df_exp_sham, tpoint_compare=change_in_altitude_tpoint)
    args.append(exp_sham_dict['altitude_change_data'])
    name_list.append('CsChrimsonxSham')
    print('Control real')
    control_real_dict = mean_change_alitude_and_CI(df_control_real, tpoint_compare=change_in_altitude_tpoint)
    args.append(control_real_dict['altitude_change_data'])
    name_list.append('ControlxLightOn')
    print('Control sham')
    control_sham_dict = mean_change_alitude_and_CI(df_control_sham, tpoint_compare=change_in_altitude_tpoint)
    args.append(control_sham_dict['altitude_change_data'])
    name_list.append('ControlxSham')
    print('Exp real')
    exp_real_dict = mean_change_alitude_and_CI(df_exp_real, tpoint_compare=change_in_altitude_tpoint)
    args.append(exp_real_dict['altitude_change_data'])
    name_list.append('CsChrimsonxLightOn')
    print('permutation tests')
    perm_p_vals = []
    name_1_list = []
    name_2_list = []
    for i in range(len(args) - 2):
        arg = args[i]
        perm_p_vals.append(permutation_test(args[-1], arg, size=1000000))
        name_1_list.append(name_list[i])
        name_2_list.append(name_list[-1])
    for i in range(len(args) - 3):
        arg = args[i]
        perm_p_vals.append(permutation_test(args[-2], arg, size=1000000))
        name_1_list.append(name_list[i])
        name_2_list.append(name_list[-2])
    reject, p_vals_corrected, _, _ = multipletests(perm_p_vals, method='holm')
    for ind, reject_bool in enumerate(reject):
        print(f'{name_1_list[ind]} vs {name_2_list[ind]} permuted holm corrected p val: {p_vals_corrected[ind]:.3e}')
    cohorts = {name: {'trajectory_ids': df['obj_id_unique'].unique().tolist(),
                             'trajectory_n': int(df['obj_id_unique'].nunique()), 'rows': len(df)}
               for name, df in [('exp_real', df_exp_real), ('exp_sham', df_exp_sham),
                                ('control_real', df_control_real), ('control_sham', df_control_sham)]}
    scalar_groups = [control_sham_dict, exp_sham_dict, control_real_dict, exp_real_dict]
    for group in scalar_groups:
        group['jitter'] = np.random.uniform(-0.1, 0.1, len(group['altitude_change_data']))
    return dict(traces={'df_exp_real_mean_CI': df_exp_real_mean_CI, 'df_exp_sham_mean_CI': df_exp_sham_mean_CI, 'df_control_real_mean_CI': df_control_real_mean_CI, 'df_control_sham_mean_CI': df_control_sham_mean_CI}, cohorts=cohorts, scalar_groups=scalar_groups, pvalues=perm_p_vals, adjusted_pvalues=p_vals_corrected, reject=reject, comparison_names=list(zip(name_1_list, name_2_list)))

