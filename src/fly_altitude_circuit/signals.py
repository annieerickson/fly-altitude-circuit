"""Stimulus detection and interpolation shared by tethered workflows.
"""

import numpy as np
from scipy import interpolate

def get_start_end_indices_stimulus_df(df, stimulus_volt, range_volts=0.25, half_volt=True, adjust_volt=False):
    """
    return arrays of start and end indices of particular stimulus volts += range volts (aka voltage sent to panels) 
    stim_df: dataframe of all phidget/behavioral/stimuli output (at imaging hz)
    stimulus_volt: value in pattern dict define above
    range_volts: upper and lower limits of acceptable variability in voltage signal (since separate 2.5 and 3 voltages, need 0.25 for no overlap)
    double_volt: if you expect the voltage to be 1/2 of the value in the pattern dictionary 
    for loom use smaller range_volts (0.075)
    """
    if half_volt==True:
        stim_volt = stimulus_volt*0.5
    else:
        stim_volt = stimulus_volt

    #some patt need to subtract 1 after 1/2 volt
    if adjust_volt==True:
        stim_volt = stim_volt -1


    lower_volt = stim_volt - range_volts
    upper_volt = stim_volt + range_volts

    #find all indices in range of stimulus vol +- range_volts
    indices = df.loc[(df['patV'] >= lower_volt) & (df['patV'] < upper_volt)].index #for close over lap cases
    # now find starts and stops 
    start_list = []
    end_list = []

    for id_num, idx in enumerate(indices):
        
        if np.isin(idx-1, indices, invert=True) and np.isin(idx+1, indices) and (id_num!=len(indices)-1): # if previous index not in list of indexes for this stimulus- start
            start_list.append(df.iloc[[idx]]['timeVec'].values[0])
        elif np.isin(idx+1, indices, invert=True) and (np.isin(idx-1, indices)): # if next index is not in list of indices for this stimulus- end
            end_list.append(df.iloc[[idx]]['timeVec'].values[0])
    
    return start_list, end_list

def get_start_end_indices_stimVolts_df(df, stimulus_volt, range_volts=0.25):
    """
    return arrays of start and end indices of particular stimulus volts (aka voltage sent to LED) += range volts 
    stim_df: dataframe of all phidget/behavioral/stimuli output (at imaging hz)
    stimulus_volt: value in stimVolt define above (2,3,4 V etc)
    range_volts: upper and lower limits of acceptable variability in voltage signal (since separate 2.5 and 3 voltages, need 0.25 for no overlap)
    """
    
    lower_volt = stimulus_volt - range_volts
    upper_volt = stimulus_volt + range_volts

    #find all indices in range of stimulus vol +- range_volts
    indices = df.loc[(df['stimVolt'] >= lower_volt) & (df['stimVolt'] < upper_volt)].index #for close over lap cases
    # now find starts and stops 
    start_list = []
    end_list = []

    for id_num, idx in enumerate(indices):
        
        if np.isin(idx-1, indices, invert=True) and np.isin(idx+1, indices) and (id_num!=len(indices)-1): # if previous index not in list of indexes for this stimulus- start
            start_list.append(df.iloc[[idx]]['timeVec'].values[0])
        elif np.isin(idx+1, indices, invert=True) and (np.isin(idx-1, indices)): # if next index is not in list of indices for this stimulus- end
            end_list.append(df.iloc[[idx]]['timeVec'].values[0])
    
    return start_list, end_list

def interpolate_trace(time_arr, data_arr, t_interp):
    """
    """
    f = interpolate.interp1d(time_arr[~np.isnan(data_arr)], data_arr[~np.isnan(data_arr)], bounds_error=False) 
    data_int = f(t_interp)
    return data_int
