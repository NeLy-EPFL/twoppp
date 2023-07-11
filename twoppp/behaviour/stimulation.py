"""
sub-module to analyse data from recordings with olfactory stimulation
"""
import os
import sys
import pickle
import glob
from scipy.ndimage import gaussian_filter1d, median_filter, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import utils2p
import utils2p.synchronization
from utils2p.synchronization import get_lines_from_h5_file, process_cam_line, process_stimulus_line, crop_lines, get_times, SyncMetadata
import utils_video.generators

from twoppp import load, utils, rois
from twoppp import plot as myplt
from twoppp.behaviour.synchronisation import reduce_during_2p_frame, get_processed_lines, reduce_mean, reduce_max, reduce_most_freq, reduce_max_bool, reduce_first_and_last_str
from twoppp.plot import videos

THR_LASER_ON = 10
THR_HEAD = -0.53
THR_HEAD_CC = -0.5
THR_THORAX_CC = -0.48
THR_THORAX = -0.45
THR_ZERO = - 1.15

STIM_LEVELS = np.array([0, 1, 5, 10, 20])
STIM_RANGES = np.array([
    [-1, 0.02],
    [0.02, 0.3],
    [0.3, 0.7],
    [0.7, 1.4],
    [1.4, 2.5]
])

def compute_background_mean(stack, N_rows=50, len_convolution=20):
    stack = utils.get_stack(stack)
    background_mean = (np.mean(stack[:, :N_rows,:], axis=(1,2)) + np.mean(stack[:, -N_rows:,:], axis=(1,2))) / 2
    thres = (np.quantile(background_mean, 0.95) + np.quantile(background_mean, 0.05)) / 2
    background_mean_fit = (background_mean > thres).astype(float) * np.mean(background_mean[background_mean > thres])
    background_mean_residual = background_mean - background_mean_fit
    background_mean_final = background_mean_fit + np.convolve(background_mean_residual, v=np.ones((len_convolution))/len_convolution, mode="same")
    return background_mean_final

def background_correct_stack(stack, N_rows=50, len_convolution=20, medfilt=(1,5,5)):
    stack = utils.get_stack(stack)
    background_mean = compute_background_mean(stack, N_rows=N_rows, len_convolution=len_convolution)
    stack_corrected = np.clip(stack - background_mean[:, np.newaxis, np.newaxis], a_min=0, a_max=None)
    stack_corrected = median_filter(stack_corrected, size=medfilt)
    return stack_corrected

def get_sync_signals_stimulation(trial_dir, sync_out_file="stim_sync.pkl", paradigm_out_file="stim_paradigm.pkl",
                                 overwrite=False, index_df=None, df_out_dir=None):
    processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
    sync_file = utils2p.find_sync_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    try:
        metadata_2p_file = utils2p.find_metadata_file(trial_dir)
        twop_exists = True
    except:
        metadata_2p_file = None
        twop_exists = False
    try:
        seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)
        cam_exists = True
    except:
        print("Warning: could not find 7cam metadata file. Will continue without")
        seven_camera_metadata_file = None
        cam_exists = False
    sync_out_file = os.path.join(processed_dir, sync_out_file)
    paradigm_out_file = os.path.join(processed_dir, paradigm_out_file)

    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    elif index_df is not None:
        assert isinstance (index_df, pd.DataFrame)

    data_read = False
    
    if os.path.isfile(paradigm_out_file) and not overwrite:
        try:
            with open(paradigm_out_file, "rb") as f:
                out_info = pickle.load(f)
            t_trig = out_info["t_cam"]
            condition_signals = out_info["condition_signals"]
            trig_laser_start = out_info["start_cam_frames"]
            trig_laser_stop = out_info["stop_cam_frames"]
            condition_list = out_info["condition_list"]
            power_signal = out_info["power_signal"]
            data_read = True
        except:
            print("Error while reading paradigm from file. Will re-compute.")

    if not data_read:
        if os.path.isfile(sync_out_file) and not overwrite:
            with open(sync_out_file, "rb") as f:
                processed_lines = pickle.load(f)
        else:
            processed_lines = get_processed_lines(
                sync_file=sync_file,
                sync_metadata_file=sync_metadata_file,
                metadata_2p_file=metadata_2p_file,
                read_cams=cam_exists,
                seven_camera_metadata_file=seven_camera_metadata_file,
                additional_lines=["LightSheetGalvoPosition", "LightSheetLaserOn", "LightSheetLaserPower"])
            with open(sync_out_file, "wb") as f:
                pickle.dump(processed_lines, f)
        
        if cam_exists:
            processed_lines_crop = {}
            for k in processed_lines.keys():
                processed_lines_crop[k] = processed_lines[k][processed_lines["Cameras"]>=0]
            processed_lines = processed_lines_crop
        else:
            pass
            # TODO: make a "Cameras" lines

        t = processed_lines["Times"]
        t = t - t[0]
        if cam_exists:
            cam = processed_lines["Cameras"]
        else:
            cam = None
        if twop_exists:
            twop = processed_lines["Frame Counter"]
        else:
            twop = None

        if "LightSheetGalvoPosition" in list(processed_lines.keys()):
            galvo_signal_raw = gaussian_filter1d(processed_lines["LightSheetGalvoPosition"], sigma=100)
        else:
            galvo_signal_raw = -2 * np.ones(processed_lines["LightSheetLaserOn"])
        if "LightSheetLaserPower" in list(processed_lines.keys()):
            power_signal_raw = gaussian_filter1d(processed_lines["LightSheetLaserPower"], sigma=100)
        else:
            power_signal_raw = np.zeros_like(processed_lines["LightSheetLaserOn"])

        if cam_exists:
            trig = cam
            t_trig = np.hstack((0, t[np.where(np.diff(cam))[0]]))  # TODO: verify!!
            trigger = "cam"
        elif twop_exists:
            trig = twop
            t_trig = t[np.where(np.diff(twop))[0]+1]  # +1 because of diff function
            trigger = "twop"
        else:
            trig = np.arange(len(galvo_signal_raw))
            t_trig = t
            trigger = "sync"

        laser_on = processed_lines["LightSheetLaserOn"] > THR_LASER_ON
        i_laser_start = np.where(np.diff(laser_on.astype(int)) == 1)[0]
        i_laser_stop = np.where(np.diff(laser_on.astype(int)) == -1)[0]
        trig_laser_start = trig[i_laser_start]
        trig_laser_stop = trig[i_laser_stop]

        N_stim = np.minimum(len(i_laser_start), len(i_laser_stop))
        galvo_list = np.zeros((N_stim))
        power_list = np.zeros((N_stim))
        condition_list = []

        thorax_signal = np.zeros_like(t_trig).astype(bool)
        head_signal = np.zeros_like(t_trig).astype(bool)
        cc_signal = np.zeros_like(t_trig).astype(bool)
        back_signal = np.zeros_like(t_trig).astype(bool)
        front_signal = np.zeros_like(t_trig).astype(bool)
        zero_signal = np.zeros_like(t_trig).astype(bool)
        power_signal = np.zeros_like(t_trig)

        for i_, (i_start, i_stop) in enumerate(zip(i_laser_start, i_laser_stop)):
            diff = i_stop - i_start
            start = int(i_start + diff/3)
            stop = int(i_start + 2*diff/3)
            cond = np.mean(galvo_signal_raw[start:stop])  # check in central third of stimulation interval
            galvo_list[i_] = cond
            power = np.mean(power_signal_raw[start:stop])
            power_list[i_] = power
            power_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = power
            if cond >= THR_THORAX_CC and cond < THR_THORAX:
                condition_list.append("thorax")
                thorax_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            elif cond < THR_THORAX_CC and cond >= THR_HEAD_CC:
                condition_list.append("cc")
                cc_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            elif cond < THR_HEAD_CC and cond > THR_HEAD:
                condition_list.append("head")
                head_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            elif cond >= THR_THORAX:
                condition_list.append("back")
                back_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            elif cond < THR_HEAD and cond >= THR_ZERO:
                condition_list.append("front")
                front_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            elif cond < THR_ZERO:
                condition_list.append("zero")
                zero_signal[trig_laser_start[i_]:trig_laser_stop[i_]] = True
            else:
                condition_list.append("error")

        condition_signals = [head_signal, cc_signal, thorax_signal, back_signal,
                             front_signal, zero_signal]
        # condition_names = ["head", "cc", "thorax", "back", "front", "zero"]

        out_info = {"t_cam": t_trig,
                    "condition_signals": condition_signals,
                    "power_signal": power_signal,
                    "start_cam_frames": trig_laser_start,
                    "stop_cam_frames": trig_laser_stop,
                    "condition_list": condition_list,
                    "trigger": trigger}
        with open(paradigm_out_file, "wb") as f:
            pickle.dump(out_info, f)

    if index_df is not None:  # TODO: make df for case when no index_df is supplied
        assert len(index_df) == len(t_trig)
        binary_cond_signal = np.zeros_like(t_trig).astype(bool)
        condition_name_signal = [""] * len(t_trig)
        for i_start, i_end, cond in zip(trig_laser_start, trig_laser_stop, condition_list):
            binary_cond_signal[i_start:i_end] = True
            for i in range(i_start, i_end):
                condition_name_signal[i] = cond
        index_df["laser_stim"] = binary_cond_signal
        index_df["laser_cond"] = condition_name_signal
        index_df["laser_power"] = power_signal
        index_df["laser_power_uW"] = get_stim_p_uW(power_signal, return_mean=False)
        index_df["laser_start"] = np.diff(np.array(binary_cond_signal).astype(int), prepend=0) == 1
        index_df["laser_stop"] = np.diff(np.array(binary_cond_signal).astype(int), prepend=0) == -1

    if df_out_dir is not None and index_df is not None:
        index_df.to_pickle(df_out_dir)

    return t_trig, condition_signals, trig_laser_start, trig_laser_stop, condition_list

def get_stim_p_uW(laser_power, return_mean=False):
    laser_power = np.nan_to_num(np.array(laser_power))
    if return_mean:
        laser_power = np.array(np.mean(laser_power))
    laser_power_uW = -1 * np.ones_like(laser_power)
    for stim_level, stim_range in zip(STIM_LEVELS, STIM_RANGES):
        meets_cond = np.logical_and(laser_power >= stim_range[0], laser_power < stim_range[1])
        laser_power_uW[meets_cond] = stim_level
    return laser_power_uW

def get_trial_stim_level(laser_power_uW, stim_start, stim_stop, fraction=0.5, laser_power_raw=None):
    stim_dur = stim_stop - stim_start
    i_start = int(stim_start + stim_dur * (1 - fraction) // 2)
    i_stop = int(stim_start + stim_dur * (1 + fraction) // 2)

    if laser_power_uW is not None:
        return reduce_most_freq(laser_power_uW[i_start:i_stop])
    else:
        return get_stim_p_uW(laser_power_raw[i_start:i_stop], return_mean=True)

# many old functions are now in:
# jonas-data-analysis-scratch/scripts/gng/moved_from_twoppp_stimulation.py
