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
from twoppp.behaviour.synchronisation import reduce_during_2p_frame, get_processed_lines, reduce_mean, reduce_max, reduce_most_freq, reduce_max_bool
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
        index_df["laser_start"] = np.diff(np.array(binary_cond_signal).astype(int), prepend=0) == 1
        index_df["laser_stop"] = np.diff(np.array(binary_cond_signal).astype(int), prepend=0) == -1

    if df_out_dir is not None and index_df is not None:
        index_df.to_pickle(df_out_dir)

    return t_trig, condition_signals, trig_laser_start, trig_laser_stop, condition_list

def get_stim_p_mW(laser_power):
    laser_power = np.mean(laser_power)
    if np.isnan(laser_power):
        laser_power = 0.0
    i_stim = int(np.where([laser_power >= stim_range[0] and laser_power < stim_range[1] for stim_range in STIM_RANGES])[0])
    return STIM_LEVELS[i_stim]

def get_trial_stim_level(laser_power_mW, stim_start, stim_stop, fraction=0.5, laser_power_raw=None):
    stim_dur = stim_stop - stim_start
    i_start = int(stim_start + stim_dur * (1 - fraction) // 2)
    i_stop = int(stim_start + stim_dur * (1 + fraction) // 2)

    if laser_power_mW is not None:
        return reduce_most_freq(laser_power_mW[i_start:i_stop])
    else:
        return get_stim_p_mW(laser_power_raw[i_start:i_stop])

beh_twop_key_map = {
    "v": ("v", reduce_mean),
    "th": ("th", reduce_mean),
    "delta_rot_lab_forward": ("v_forw", reduce_mean),
    "delta_rot_lab_side": ("v_side", reduce_mean),
    "delta_lab_rot_turn": ("v_turn", reduce_mean),
    "laser_stim": ("laser_stim", reduce_max_bool),
    "laser_power": ("laser_power_mW", get_stim_p_mW),
    "laser_start": ("laser_start", reduce_max_bool),
    "laser_stop": ("laser_stop", reduce_max_bool),
    # "": ("", reduce_mean),
}

def get_beh_info_to_twop_df(beh_df, twop_df, twop_df_out_dir=None, key_map=beh_twop_key_map):
    if isinstance(beh_df, str) and os.path.isfile(beh_df):
        beh_df = pd.read_pickle(beh_df)
    assert isinstance (beh_df, pd.DataFrame)
    if isinstance(twop_df, str) and os.path.isfile(twop_df):
        twop_df = pd.read_pickle(twop_df)
    assert isinstance (twop_df, pd.DataFrame)

    twop_index = beh_df.twop_index.values
    for beh_key, (twop_key, red_function) in key_map.items():
        try:
            signal = reduce_during_2p_frame(
                twop_index=twop_index,
                values=beh_df[beh_key],
                function=red_function)
        except KeyError:
            Warning(f"Could not find key {beh_key} in behaviour_df. Will continue.")
            continue
        twop_df.loc[:, twop_key] = np.zeros((len(twop_df), 1), dtype=signal.dtype)
        twop_df.loc[:len(signal), twop_key] = signal
        
    if twop_df_out_dir is not None:
        twop_df.to_pickle(twop_df_out_dir)
    return twop_df

def add_beh_state_to_twop_df(twop_df, twop_df_out_dir=None):
    if isinstance(twop_df, str) and os.path.isfile(twop_df):
        twop_df = pd.read_pickle(twop_df)
    assert isinstance (twop_df, pd.DataFrame)

    def backwards_walking(v_forw, thres_back=-0.25, winsize=4): 
        back_walk = gaussian_filter1d(v_forw, sigma=5) < thres_back
        back_walk = np.logical_and(np.convolve(back_walk, np.ones(winsize)/winsize, mode="same") >= 0.75, back_walk)
        return back_walk

    def walking(v_forw, thres_walk=0.75, winsize=4): 
        walk = gaussian_filter1d(v_forw, sigma=5) > thres_walk
        walk = np.logical_and(np.convolve(walk, np.ones(winsize)/winsize, mode="same") >= 0.75, walk)
        return walk

    def resting(v, thres_rest=0.5, winsize=8, walk=None, back=None): 
        rest = gaussian_filter1d(v, sigma=5) <= thres_rest
        if walk is not None:
            rest = np.logical_and(rest, np.logical_not(walk))
        if back is not None:
            rest = np.logical_and(rest, np.logical_not(back))
        rest = np.logical_and(np.convolve(rest, np.ones(winsize)/winsize, mode="same") >= 0.75, rest)
        return rest

    twop_df["back"] = backwards_walking(twop_df.v_forw.values).astype(int)
    twop_df["walk"] = walking(twop_df.v_forw.values).astype(int)
    twop_df["rest"] = resting(twop_df.v_forw.values, walk=twop_df["walk"].values, back=twop_df["back"].values).astype(int)
    twop_df["beh_catvar"] = twop_df["rest"].values + 2 * twop_df["walk"].values + 4 * twop_df["back"].values

    if twop_df_out_dir is not None:
        twop_df.to_pickle(twop_df_out_dir)
    return twop_df

def get_triggered_avg_videos(green_stacks, twop_dfs, video_dir, stim_dur_s=5, walk_rest=False, pre_stim_s=2, pre_stim_baseline_dur_s=1, appendix="", stand="exclude", stand_post_excl_s=5, highp=False):
    assert len(green_stacks) == len(twop_dfs)
    N_trials = len(twop_dfs)
    N_frames_df = np.zeros((len(twop_dfs)))
    for i_df, twop_df in enumerate(twop_dfs):
        if isinstance(twop_df, str) and os.path.isfile(twop_df):
            twop_dfs[i_df] = pd.read_pickle(twop_df)
        assert isinstance (twop_dfs[i_df], pd.DataFrame)
        N_frames_df[i_df] = len(twop_dfs[i_df])

    for i_stack, stack in enumerate(green_stacks):
        if isinstance(stack, str) and os.path.isfile(stack):
            stack = utils.get_stack(stack)
        assert isinstance(stack, np.ndarray)
        if len(stack) == N_frames_df[i_stack] + 60:
            stack = stack[30:-30]
        assert len(stack) == N_frames_df[i_stack]
        if stand == "all":
            stack_mean = np.mean(stack, dtype=np.float32, axis=0)
            stack_std = np.std(stack, dtype=np.float32, axis=0)
            green_stacks[i_stack] = (stack - stack_mean) / stack_std
            del stack_mean, stack_std
        else:
            print(f"filtering stack {i_stack}")
            # green_stacks[i_stack] = stack.astype(np.float32)
            green_stacks[i_stack] = gaussian_filter(median_filter(stack.astype(np.float32), size=(1,3,3)), sigma=(1,1,1))
    del stack

    dt = np.mean(np.diff(twop_dfs[0].t))
    fs = 1 / dt
    N_samples_stim = int(fs * stim_dur_s)
    N_samples_pre = int(fs * pre_stim_s)
    N_samples = N_samples_stim*2 + N_samples_pre
    # stim_t = (np.arange(N_samples) - N_samples_pre) / N_samples_stim * stim_dur_s

    twop_stim_starts = [np.argwhere(twop_df.laser_start.values==True).flatten() for twop_df in twop_dfs]
    twop_stim_stops = [np.argwhere(twop_df.laser_stop.values==True).flatten() for twop_df in twop_dfs]
    N_stims_per_trial = [len(twop_stim_start) for twop_stim_start in twop_stim_starts]
    # assert len(np.unique(N_stims_per_trial)) == 1  # TODO: adapt to different # of stims per trial
    N_stims_per_trial = np.max(N_stims_per_trial)

    if stand == "exclude":
        # stand_post_excl_s
        # laser_stim = pd.concat([twop_df["laser_stim"] for twop_df in twop_dfs])["laser_stim"].values
        stand_post_excl_samples = int(stand_post_excl_s * fs)
        stack_mean = np.zeros((green_stacks[0].shape[1:]), dtype=np.float32)
        N_samples_mean = 0
        no_laser_stims = []
        for stack, trial_stim_starts, trial_stim_stops in zip(green_stacks, twop_stim_starts, twop_stim_stops):
            no_laser_stim = np.ones(stack.shape[0], dtype=bool)
            for stim_start, stim_stop in zip(trial_stim_starts, trial_stim_stops):
                no_laser_stim[stim_start:stim_start+int(fs*5)+stand_post_excl_samples] = False  # stim_stop
            no_laser_stims.append(no_laser_stim)
            N_samples_mean += np.sum(no_laser_stim.astype(int))
            stack_mean += np.sum(stack[no_laser_stim,:,:], axis=0, dtype=np.float32)
        stack_mean /= N_samples_mean

        stack_std = np.zeros_like(stack_mean, dtype=np.float32)
        for i_trial, (stack, no_laser_stim) in enumerate(zip(green_stacks, no_laser_stims)):
            stack = stack.astype(np.float32) - stack_mean
            green_stacks[i_trial] = stack
            stack_std += np.sum(np.square(stack[no_laser_stim,:,:]), axis=0, dtype=np.float32)

        stack_std /= N_samples_mean
        stack_std = np.sqrt(stack_std)

        for i_trial, stack in enumerate(green_stacks):
            stack /= stack_std
            green_stacks[i_trial] = stack

    try:
        twop_stim_levels = [[get_trial_stim_level(twop_df["laser_power_mW"].values, trial_start, trial_stop)
            for trial_start, trial_stop in zip(trial_stim_starts, trial_stim_stops)]
                for twop_df, trial_stim_starts, trial_stim_stops in zip(twop_dfs, twop_stim_starts, twop_stim_stops)]
    except KeyError:
        twop_stim_levels = [[get_trial_stim_level(None, trial_start, trial_stop, laser_power_raw=twop_df["laser_power"].values)
            for trial_start, trial_stop in zip(trial_stim_starts, trial_stim_stops)]
                for twop_df, trial_stim_starts, trial_stim_stops in zip(twop_dfs, twop_stim_starts, twop_stim_stops)]

    twop_i_stim_levels = [[int(np.where(STIM_LEVELS == stim_level)[0]) for stim_level in _] for _ in twop_stim_levels]

    max_stim_per_level = N_stims_per_trial * N_trials // (len(STIM_LEVELS) -1)
    stim_trig_stacks = np.zeros((N_samples, green_stacks[0].shape[1], green_stacks[0].shape[2], len(STIM_LEVELS), max_stim_per_level), dtype=np.float32)

    i_stims_per_level = np.zeros(len(STIM_LEVELS)).astype(int)

    for green_z, trial_stim_start, trial_i_stim_level in zip(green_stacks, twop_stim_starts, twop_i_stim_levels):
        for this_stim_start, this_i_stim_level in tqdm(zip(trial_stim_start, trial_i_stim_level)):
            i_start = np.maximum(0, this_stim_start-N_samples_pre)
            i_start_out = np.maximum(0, -1*(this_stim_start-N_samples_pre))

            signals = green_z[i_start:this_stim_start+2*N_samples_stim,:,:]
            stim_trig_stacks[i_start_out:,:,:,this_i_stim_level, i_stims_per_level[this_i_stim_level]] = signals
            i_stims_per_level[this_i_stim_level] += 1
    
    del green_stacks, signals
    # assert all(i_stims_per_level[1:] == max_stim_per_level) TODO: adapt to different # of stimulus repetitions
    stim_trig_stacks_mean = np.mean(stim_trig_stacks, axis=-1)
    # del stim_trig_stacks
    if highp:
        highp_stim_trig_stacks_mean = np.mean(stim_trig_stacks_mean[:,:,:,-2:], axis=-1)
        del stim_trig_stacks_mean
        if stand == "all":
            highp_stim_trig_stacks_mean_filt = gaussian_filter(median_filter(highp_stim_trig_stacks_mean, size=(1,3,3)), sigma=(1,2,2))
        else:
            highp_stim_trig_stacks_mean_filt = highp_stim_trig_stacks_mean
        del highp_stim_trig_stacks_mean

        pre_stim = [N_samples_pre-int(pre_stim_baseline_dur_s*fs),N_samples_pre]
        highp_baseline = np.mean(highp_stim_trig_stacks_mean_filt[pre_stim[0]:pre_stim[1]], axis=0)
        highp_stim_trig_stacks_mean_filt_rel = highp_stim_trig_stacks_mean_filt - highp_baseline
        del highp_stim_trig_stacks_mean_filt, highp_baseline
        
        if stand=="all":
            generator = videos.generator_dff(stack=highp_stim_trig_stacks_mean_filt_rel[:,80:-80,40:-40],
                                            vmin=-0.5, vmax=0.5,  # -0.5, vmax=0.5,
                                            text="p=10/20uW",
                                            colorbarlabel="std")
        elif stand=="exclude":
            generator = videos.generator_dff(stack=highp_stim_trig_stacks_mean_filt_rel[:,80:-80,40:-40],
                                            vmin=-2, vmax=2,  # -0.5, vmax=0.5,
                                            text="p=10/20uW",
                                            colorbarlabel="std")
        generator = videos.stimulus_dot_generator(generator, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim])
        videos.make_video(os.path.join(video_dir, f"neural_stim_aligned_highp{appendix}.mp4"), generator, fps=fs, n_frames=N_samples)

        del highp_stim_trig_stacks_mean_filt_rel
    else:
        if stand == "all":
            stim_trig_stacks_mean_filt = gaussian_filter(median_filter(stim_trig_stacks_mean[:,:,:,1:], size=(1,3,3,1)), sigma=(1,2,2,0))
        else:
            # stim_trig_stacks_mean_filt = gaussian_filter(median_filter(stim_trig_stacks_mean[:,:,:,1:], size=(1,3,3,1)), sigma=(1,1,1,0))
            stim_trig_stacks_mean_filt = stim_trig_stacks_mean[:,:,:,1:]
        del stim_trig_stacks_mean

        pre_stim = [N_samples_pre-int(pre_stim_baseline_dur_s*fs),N_samples_pre]
        baseline = np.mean(stim_trig_stacks_mean_filt[pre_stim[0]:pre_stim[1]], axis=0)
        stim_trig_stacks_mean_filt_rel = stim_trig_stacks_mean_filt - baseline
        del stim_trig_stacks_mean_filt, baseline

        if stand=="all":
            vmax = 0.5
        elif stand=="exclude":
            vmax = 3  # 2
        else:
            vmax = 1
        generators = [
            videos.generator_dff(stack=stim_trig_stacks_mean_filt_rel[:,80:-80,40:-40,i],
                                vmin=-vmax, vmax=vmax,
                                text=label,
                                show_colorbar=show_colorbar,
                                colorbarlabel="std")
            for i, (label, show_colorbar) in enumerate(zip(["1uW", "5uW", "10uW", "20uW"],[False, False, False, True]))
        ]
        generator = utils_video.generators.stack(generators, axis=1)
        generator = videos.stimulus_dot_generator(generator, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim])
        videos.make_video(os.path.join(video_dir, f"neural_stim_aligned_allp{appendix}.mp4"), generator, fps=fs, n_frames=N_samples)
        del stim_trig_stacks_mean_filt_rel

    if walk_rest:
        rest_vs_walk = np.zeros((N_trials, N_stims_per_trial), dtype=int)
        # stim_trig_stacks = np.zeros((N_samples_stim*3, green_stacks[0].shape[1], green_stacks[0].shape[2], len(STIM_LEVELS), max_stim_per_level), dtype=np.float32)
        i_stims_per_level_2 = np.zeros(len(STIM_LEVELS)).astype(int)
        n_walk_rest_per_level = np.zeros((len(STIM_LEVELS),2)).astype(int)
        stim_trig_stacks_mean_walk_rest = np.zeros(stim_trig_stacks.shape[:4]+(2,), dtype=np.float32)
        for i_trial, (trial_stim_start, twop_df, trial_i_stim_level) in enumerate(zip(twop_stim_starts, twop_dfs, twop_i_stim_levels)):
            for i_stim, (this_stim_start, this_i_stim_level) in enumerate(zip(trial_stim_start, trial_i_stim_level)):
                this_rest = (twop_df.beh_catvar.values[this_stim_start-int(pre_stim_baseline_dur_s*fs):this_stim_start] == 1).astype(int)
                this_walk = (twop_df.beh_catvar.values[this_stim_start-int(pre_stim_baseline_dur_s*fs):this_stim_start] == 2).astype(int)
                if np.sum(this_walk) > np.sum(this_rest):
                    rest_vs_walk[i_trial, i_stim] = 1
                    stim_trig_stacks_mean_walk_rest[:,:,:,this_i_stim_level,1] += stim_trig_stacks[:,:,:,this_i_stim_level,i_stims_per_level_2[this_i_stim_level]]
                    stim_trig_stacks_mean_walk_rest[this_i_stim_level,1] += 1
                    n_walk_rest_per_level[this_i_stim_level,1] += 1
                else:
                    stim_trig_stacks_mean_walk_rest[:,:,:,this_i_stim_level,0] += stim_trig_stacks[:,:,:,this_i_stim_level,i_stims_per_level_2[this_i_stim_level]]
                    stim_trig_stacks_mean_walk_rest[this_i_stim_level,0] += 1
                    n_walk_rest_per_level[this_i_stim_level,0] += 1
                i_stims_per_level_2[this_i_stim_level] += 1
        
        del stim_trig_stacks
        highp_stim_trig_stacks_mean = np.sum(stim_trig_stacks_mean_walk_rest[:,:,:,-2:,:], axis=-2) / np.sum(n_walk_rest_per_level[-2:,:], axis=-2)
        del stim_trig_stacks_mean_walk_rest
        highp_stim_trig_stacks_mean_walk = highp_stim_trig_stacks_mean[:,:,:,1]
        highp_stim_trig_stacks_mean_rest = highp_stim_trig_stacks_mean[:,:,:,0]
        del highp_stim_trig_stacks_mean

        if stand == "all":
            highp_stim_trig_stacks_mean_walk_filt = gaussian_filter(median_filter(highp_stim_trig_stacks_mean_walk, size=(1,3,3)), sigma=(1,2,2))
            del highp_stim_trig_stacks_mean_walk
            highp_stim_trig_stacks_mean_rest_filt = gaussian_filter(median_filter(highp_stim_trig_stacks_mean_rest, size=(1,3,3)), sigma=(1,2,2))
            del highp_stim_trig_stacks_mean_rest
        else:
            highp_stim_trig_stacks_mean_walk_filt = highp_stim_trig_stacks_mean_walk
            del highp_stim_trig_stacks_mean_walk
            highp_stim_trig_stacks_mean_rest_filt = highp_stim_trig_stacks_mean_rest
            del highp_stim_trig_stacks_mean_rest

        generator = videos.generator_dff(stack=highp_stim_trig_stacks_mean_walk_filt[:,80:-80,40:-40],
                                                vmin=-0.5, vmax=0.5,
                                                text="p=10/20uW",
                                                colorbarlabel="std")
        generator = videos.stimulus_dot_generator(generator, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim])
        videos.make_video(os.path.join(fly_dir, load.PROCESSED_FOLDER, "neural_stim_aligned_highp_walk.mp4"), generator, fps=fs, n_frames=N_samples)

        generator = videos.generator_dff(stack=highp_stim_trig_stacks_mean_rest_filt[:,80:-80,40:-40],
                                                vmin=-0.5, vmax=0.5,
                                                text="p=10/20uW",
                                                colorbarlabel="std")
        generator = videos.stimulus_dot_generator(generator, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim])
        videos.make_video(os.path.join(fly_dir, load.PROCESSED_FOLDER, "neural_stim_aligned_highp_rest.mp4"), generator, fps=fs, n_frames=N_samples)

        del highp_stim_trig_stacks_mean_rest_filt,highp_stim_trig_stacks_mean_walk_filt

def get_triggered_beh_neural_plot(twop_dfs, plot_name=None, plot_out_dir=None, stim_dur_s=5, pre_stim_baseline_dur_s=1):
    N_trials = len(twop_dfs)
    N_frames_df = np.zeros((len(twop_dfs)))
    for i_df, twop_df in enumerate(twop_dfs):
        if isinstance(twop_df, str) and os.path.isfile(twop_df):
            twop_dfs[i_df] = pd.read_pickle(twop_df)
        assert isinstance (twop_dfs[i_df], pd.DataFrame)
        N_frames_df[i_df] = len(twop_dfs[i_df])

    twop_df = pd.concat(twop_dfs)

    neurons = utils.standardise(twop_df.filter(regex="neuron").filter(regex="^(?!.*?denoised)").values)[:,:]
    dffs, f0s = rois.get_dff_from_traces(twop_df.filter(regex="neuron").filter(regex="^(?!.*?denoised)").values[:,:], return_f0=True)
    _, fmaxs = rois.get_dff_from_traces(-dffs, return_f0=True)
    norm_dffs = dffs / -fmaxs

    # neurons_denoised = utils.standardise(twop_df.filter(regex="neuron_denoised").values)[:,:]
    # dffs_denoised, f0s_denoised = rois.get_dff_from_traces(twop_df.filter(regex="neuron_denoised").values[:,:], return_f0=True)
    N_neurons = neurons.shape[1]

    dt = np.mean(np.diff(twop_dfs[0].t))
    fs = 1 / dt
    N_samples_stim = int(fs * stim_dur_s)
    stim_t = (np.arange(3*N_samples_stim) - N_samples_stim) / N_samples_stim * stim_dur_s

    twop_stim_starts = np.argwhere(twop_df.laser_start.values==True).flatten()
    twop_stim_stops = np.argwhere(twop_df.laser_stop.values==True).flatten()
    N_stims_per_trial = len(twop_stim_starts)

    try:
        twop_stim_levels = [get_trial_stim_level(twop_df["laser_power_mW"].values, trial_start, trial_stop)
            for trial_start, trial_stop in zip(twop_stim_starts, twop_stim_stops)]
    except KeyError:
        twop_stim_levels = [get_trial_stim_level(None, trial_start, trial_stop, laser_power_raw=twop_df["laser_power"].values)
            for trial_start, trial_stop in zip(twop_stim_starts, twop_stim_stops)]

    twop_i_stim_levels = [int(np.where(STIM_LEVELS == stim_level)[0]) for stim_level in twop_stim_levels]

    pre_stim = [N_samples_stim-int(pre_stim_baseline_dur_s*fs),N_samples_stim]
    N_stim = 80  # len(twop_stim_starts) TODO: FIX THIS
    beh_responses = np.zeros((3*N_samples_stim, 4, len(STIM_LEVELS)))
    v_responses = np.zeros((3*N_samples_stim, len(STIM_LEVELS), N_stim//4))
    neuron_responses = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), N_stim//4))
    dff_responses = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), N_stim//4))
    norm_dff_responses = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), N_stim//4))
    neuron_responses_per_beh = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), 2, N_stim//4))
    dff_responses_per_beh = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), 2, N_stim//4))
    norm_dff_responses_per_beh = np.zeros((3*N_samples_stim, N_neurons, len(STIM_LEVELS), 2, N_stim//4))
    N_per_stim_level = np.zeros(len(STIM_LEVELS)).astype(int)
    N_per_stim_level_per_beh = np.zeros((len(STIM_LEVELS),2)).astype(int)

    for i_stim, (stim_start, i_stim_level) in enumerate(zip(twop_stim_starts, twop_i_stim_levels)):
        zero_correct = -1 * np.minimum(0, stim_start-N_samples_stim)
        beh_responses[zero_correct:, 0, i_stim_level] += (twop_df.beh_catvar.values[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim] == 0).astype(int)
        this_rest = (twop_df.beh_catvar.values[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim] == 1).astype(int)
        beh_responses[zero_correct:, 1, i_stim_level] += this_rest
        this_walk = (twop_df.beh_catvar.values[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim] == 2).astype(int)
        beh_responses[zero_correct:, 2, i_stim_level] += this_walk
        beh_responses[zero_correct:, 3, i_stim_level] += (twop_df.beh_catvar.values[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim] == 4).astype(int)
        v_responses[zero_correct:, i_stim_level, N_per_stim_level[i_stim_level]] = twop_df.v_forw.values[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim]
        this_neuron = neurons[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim,:]
        neuron_responses[zero_correct:,:,i_stim_level, N_per_stim_level[i_stim_level]] = this_neuron
        this_dff = dffs[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim,:]
        dff_responses[zero_correct:,:,i_stim_level, N_per_stim_level[i_stim_level]] = this_dff
        this_norm_dff = norm_dffs[stim_start-N_samples_stim+zero_correct:stim_start+2*N_samples_stim,:]
        norm_dff_responses[zero_correct:,:,i_stim_level, N_per_stim_level[i_stim_level]] = this_norm_dff
        N_per_stim_level[i_stim_level] += 1
        
        if np.sum(this_rest[pre_stim[0]:pre_stim[1]]) > np.sum(this_walk[pre_stim[0]:pre_stim[1]]):
            # rest before
            neuron_responses_per_beh[zero_correct:,:,i_stim_level, 0, N_per_stim_level_per_beh[i_stim_level,0]] = this_neuron
            dff_responses_per_beh[zero_correct:,:,i_stim_level, 0, N_per_stim_level_per_beh[i_stim_level,0]] = this_dff
            norm_dff_responses_per_beh[zero_correct:,:,i_stim_level, 0, N_per_stim_level_per_beh[i_stim_level,0]] = this_norm_dff
            N_per_stim_level_per_beh[i_stim_level,0] += 1
        else: # walk before
            neuron_responses_per_beh[zero_correct:,:,i_stim_level, 1, N_per_stim_level_per_beh[i_stim_level,1]] = this_neuron
            dff_responses_per_beh[zero_correct:,:,i_stim_level, 1, N_per_stim_level_per_beh[i_stim_level,1]] = this_dff
            norm_dff_responses_per_beh[zero_correct:,:,i_stim_level, 1, N_per_stim_level_per_beh[i_stim_level,1]] = this_norm_dff
            N_per_stim_level_per_beh[i_stim_level,1] += 1
        
    beh_responses /= (N_stim // 4)

    pre_stim_f0 = np.mean(neuron_responses[pre_stim[0]:pre_stim[1],:,:,:], axis=0)
    neuron_rel_responses = neuron_responses - pre_stim_f0

    pre_stim_f0_per_beh = np.mean(neuron_responses_per_beh[pre_stim[0]:pre_stim[1],:,:,:,:], axis=0)
    neuron_rel_responses_per_beh = neuron_responses_per_beh - pre_stim_f0_per_beh

    fig, axs = plt.subplots(3,2,figsize=(6,6), sharex=True)  # , sharey=True)

    ax = axs[0,1]
    for i_level, level_color in enumerate([myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED]):
        myplt.plot_mu_sem(mu=gaussian_filter1d(np.mean(v_responses[:,i_level+1,:], axis=1), sigma=1),
                        err=gaussian_filter1d(utils.conf_int(v_responses[:,i_level+1,:], axis=1), sigma=1),
                        x=stim_t,
                        label=f"{STIM_LEVELS[i_level+1]} uW",
                        ax=ax,
                        color=level_color,
                        linewidth=2
                        )
    # ax.legend(frameon=False, bbox_to_anchor=(-0.9,0.9), fontsize=12)
    ax.axhline(y=0, color='k', linewidth=2)
    ax.set_ylabel(r"$v_{||}$ (mm/s)", fontsize=16)

    axs[0,0].axis("off")

    for i_level, (ax, level_color) in enumerate(zip(axs.flatten()[2:],
                                                    [myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED])):
        
        for i_beh, (label, color) in enumerate(zip(["undefined", "rest", "forw walk", "back walk"],
                                                    [myplt.DARKGRAY, myplt.DARKBLUE, myplt.DARKRED, myplt.DARKGREEN])):
            ax.plot(stim_t, gaussian_filter1d(beh_responses[:,i_beh, i_level+1], sigma=1), label=label,color=color, linewidth=2)
        ax.set_title(f"p = {STIM_LEVELS[i_level+1]} uW", color=level_color, fontsize=16)
        ax.set_ylim([-0.05,1.05])
            
    for i_ax, ax in enumerate(axs.flatten()[1:]):
        ax.set_xlim([-5,10])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward',4))
        ax.tick_params(width=2.5)
        ax.tick_params(length=5)
        ax.tick_params(labelsize=16)
        ax.set_xlabel("t (s)", fontsize=16)
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                            x=stim_t, ax=ax, colors=[myplt.WHITE, myplt.BLACK])
        # if i_ax == 4:
        #     ax.legend(frameon=False, bbox_to_anchor=(-0.9,0.9), fontsize=12)

    fig.tight_layout()
    
    if plot_out_dir is not None and os.path.isdir(plot_out_dir):
        fig.savefig(os.path.join(plot_out_dir, "beh_responses.pdf"), dpi=300)
    


    fig, axs = plt.subplots(14,5,figsize=(9.5,15), sharex=True, sharey=True)
    colours = [myplt.BLACK, myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED]
    for i_neuron, ax in enumerate(axs.flatten()):
        if i_neuron >= N_neurons:
            ax.axis("off")
            continue
        # this_neuron_responses = neuron_responses[:,i_neuron,:,:]
        this_neuron_responses = neuron_rel_responses[:,i_neuron,:,:]  # sort_idx_max_CO2_resp
        # this_neuron_responses = dff_responses[:,i_neuron,:,:] 
        # this_neuron_responses = norm_dff_responses[:,i_neuron,:,:] 
        
        for i_level, level_color in enumerate([myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED]):
            signals = this_neuron_responses[:,i_level+1,:]
            myplt.plot_mu_sem(mu=gaussian_filter1d(np.mean(signals, axis=1), sigma=3),
                            err=gaussian_filter1d(utils.conf_int(signals, axis=1), sigma=3),
                            x=stim_t,
                            label=f"{STIM_LEVELS[i_level+1]} uW",
                            ax=ax,
                            color=level_color,
                            linewidth=1,
                            alpha=0.2
                            )
            # if i_level == 3 and np.mean(signals[int(fs*6):int(fs*9), :]) > 0.5:
            #     ax.plot([1,4], [-1,-1], color=myplt.DARKPINK, linewidth=10)
            # elif i_level == 3 and np.mean(signals[int(fs*6):int(fs*9), :]) < -0.5:
            #     ax.plot([1,4], [1,1], color=myplt.DARKCYAN, linewidth=10)
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                            x=stim_t, ax=ax, colors=[myplt.WHITE, myplt.BLACK])
        # ax.legend(frameon=False, bbox_to_anchor=(-0.9,0.9), fontsize=12)
        ax.axhline(y=0, color='k', linewidth=1)
        ax.set_ylabel(i_neuron, fontsize=16, rotation=0, labelpad=15)
        # ax.set_title(f"neuron {i_neuron}", fontsize=16)
        if i_neuron >= 60:
            ax.set_xlabel("t (s)", fontsize=16)
        
        ax.set_xlim([-5,10])
        ax.set_ylim([-1,1])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # ax.spines["left"].set_linewidth(2)
        # ax.spines["bottom"].set_linewidth(2)
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward',4))
        # ax.tick_params(width=2.5)
        # ax.tick_params(length=5)
        ax.tick_params(labelsize=12)

    fig.suptitle(f"GNG DNs responding to MDN stimulation. {plot_name}")  
    fig.tight_layout()

    if plot_out_dir is not None and os.path.isdir(plot_out_dir):
        fig.savefig(os.path.join(plot_out_dir, "neural_responses_alpha.pdf"), dpi=300)


    fig, axs = plt.subplots(13,5,figsize=(9.5,13), sharex=True, sharey=True)
    colours = [myplt.BLACK, myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED]

    to_plot = neuron_responses_per_beh
    # norm_dff_responses_per_beh  
    # neuron_rel_responses_per_beh  # neuron_responses_per_beh

    for i_neuron, ax in enumerate(axs.flatten()):
        if i_neuron >= N_neurons - 1:
            ax.axis("off")
            continue
        this_neuron_responses = to_plot[:,i_neuron,:,:,:]
        
        N_avg3_rest = N_per_stim_level_per_beh[3,0]
        N_avg4_rest = N_per_stim_level_per_beh[4,0]
        signals_rest = np.zeros((3*N_samples_stim, N_avg3_rest+N_avg4_rest))
        signals_rest[:,:N_avg3_rest] = this_neuron_responses[:,3,0,:N_avg3_rest]
        signals_rest[:,N_avg3_rest:N_avg3_rest+N_avg4_rest] = this_neuron_responses[:,4,0,:N_avg4_rest]
        
        N_avg3_walk = N_per_stim_level_per_beh[3,1]
        N_avg4_walk = N_per_stim_level_per_beh[4,1]
        signals_walk = np.zeros((3*N_samples_stim, N_avg3_walk+N_avg4_walk))
        signals_walk[:,:N_avg3_walk] = this_neuron_responses[:,3,1,:N_avg3_walk]
        signals_walk[:,N_avg3_walk:N_avg3_walk+N_avg4_walk] = this_neuron_responses[:,4,1,:N_avg4_walk]
        
        
        myplt.plot_mu_sem(mu=gaussian_filter1d(np.mean(signals_rest, axis=1), sigma=3),
                        err=gaussian_filter1d(utils.conf_int(signals_rest, axis=1), sigma=3),
                        x=stim_t,
                        label=f"rest before",
                        ax=ax,
                        color=myplt.DARKBLUE,
                        linewidth=1,
                        alpha=0.2
                        )
        myplt.plot_mu_sem(mu=gaussian_filter1d(np.mean(signals_walk, axis=1), sigma=3),
                        err=gaussian_filter1d(utils.conf_int(signals_walk, axis=1), sigma=3),
                        x=stim_t,
                        label=f"walk before",
                        ax=ax,
                        color=myplt.DARKRED,
                        linewidth=1,
                        alpha=0.2
                        )
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                            x=stim_t, ax=ax, colors=[myplt.WHITE, myplt.BLACK])
        if np.maximum(np.mean(signals_walk[int(fs*6):int(fs*9), :]), np.mean(signals_rest[int(fs*6):int(fs*9), :])) > .5:
                ax.plot([1,4], [-1,-1], color=myplt.DARKPINK, linewidth=10)
        elif np.minimum(np.mean(signals_walk[int(fs*6):int(fs*9), :]),np.mean(signals_rest[int(fs*6):int(fs*9), :])) < -.5:
            ax.plot([1,4], [1,1], color=myplt.DARKCYAN, linewidth=10)
        
        
        # ax.legend(frameon=False, bbox_to_anchor=(-0.9,0.9), fontsize=12)
        ax.axhline(y=0, color='k', linewidth=1)
        ax.set_ylabel(i_neuron, fontsize=16, rotation=0, labelpad=15)
        # ax.set_title(f"neuron {i_neuron}", fontsize=16)
        if i_neuron >= 60:
            ax.set_xlabel("t (s)", fontsize=16)
        
        ax.set_xlim([-5,10])
        ax.set_ylim([-1,1])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # ax.spines["left"].set_linewidth(2)
        # ax.spines["bottom"].set_linewidth(2)
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward',4))
        # ax.tick_params(width=2.5)
        # ax.tick_params(length=5)
        ax.tick_params(labelsize=12)

    fig.suptitle(f"GNG DNs responding to MDN stimulation after walking (red) or resting (blue). {plot_name}")  
    fig.tight_layout()

    if plot_out_dir is not None and os.path.isdir(plot_out_dir):
        fig.savefig(os.path.join(plot_out_dir, "neural_responses_walk_rest_alpha.pdf"), dpi=300)

def make_behaviour_grid_video(trial_dirs, fps=100, pre_stim_s=5, stim_s=5, post_stim_s=5, appendix=""):
    beh_dfs = utils.list_join(trial_dirs, [load.PROCESSED_FOLDER, "beh_df.pkl"])
    N_trials = len(beh_dfs)
    N_frames_df = np.zeros((len(beh_dfs)))
    for i_df, beh_df in enumerate(beh_dfs):
        if isinstance(beh_df, str) and os.path.isfile(beh_df):
            beh_dfs[i_df] = pd.read_pickle(beh_df)
        assert isinstance (beh_dfs[i_df], pd.DataFrame)
        N_frames_df[i_df] = len(beh_dfs[i_df])


    video_dirs = utils.list_join(trial_dirs, ["behData", "images", "camera_5.mp4"])
    beh_stim_starts = [np.argwhere(beh_df.laser_start.values==True).flatten() for beh_df in beh_dfs]
    start_frames = [beh_stim_start[:8]-fps*pre_stim_s for beh_stim_start in beh_stim_starts]
    myplt.videos.make_behaviour_grid_video(
        video_dirs=video_dirs,
        start_frames=start_frames,
        N_frames=(pre_stim_s+stim_s+post_stim_s)*fps,
        stim_range=[int(pre_stim_s*fps),int((pre_stim_s+stim_s)*fps)],
        out_dir=os.path.join(os.path.dirname(trial_dirs[0]), load.PROCESSED_FOLDER),
        video_name=f"beh_stim_aligned{appendix}.mp4",
        frame_rate=fps
    )

if __name__ == "__main__":
    """
    date_dir = os.path.join(load.NAS2_DIR_JB, "220602_MDN3xCsChr")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    all_trial_dirs = load.get_trials_from_fly(fly_dirs)
    trial_dir = all_trial_dirs[0][2]

    get_sync_signals_stimulation(trial_dir, sync_out_file="stim_sync.pkl", paradigm_out_file="stim_paradigm.pkl",
                               overwrite=False, index_df=None, df_out_dir=None)
    """

    fly_dirs = [
        # "/mnt/nas2/JB/220721_DfdxGCaMP6s_tdTom/Fly2",
        # "/mnt/nas2/JB/220818_DfdxGCaMP6s_tdTom/Fly2",
        # "/mnt/nas2/JB/220920_DfdxGCaMP6s_MDN3xCsChrimson/Fly1",
        # "/mnt/nas2/JB/220921_DfdxGCaMP6s_MDN3xCsChrimson/Fly1",
        # "/mnt/nas2/JB/221018_DfdxGCaMP6s_tdTom_MDN3xCsChrimson/Fly1",
        # "/mnt/nas2/JB/221028_DfdxGCaMP6s_tdTom_MDN3xCsChrimson/Fly1",
        # os.path.join(load.NAS2_DIR_JB, "221101_DfdxGCaMP6s_tdTom_CsChrimson_PR", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221109_DfdxGCaMP6s_tdTom_CsChrimsonxPR", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221110_DfdxGCaMP6s_tdTom_CsChrimsonxPR", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221115_DfdxGCaMP6s_tdTom_CsChrimsonxPR", "Fly1"),  # some problem
        # os.path.join(load.NAS2_DIR_JB, "221117_DfdxGCaMP6s_tdTom_DNP9xCsChrimson", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221129_DfdxGCaMP6s_tdTom_DNp09xCsChrimson", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221129_DfdxGCaMP6s_tdTom_DNp09xCsChrimson", "Fly2"),
        # os.path.join(load.NAS2_DIR_JB, "221201_DfdxGCaMP6s_tdTom_MDN3xCsChrimson", "Fly1"),
        # os.path.join(load.NAS2_DIR_JB, "221213_ScrxGCaMP6s_MDN3xCsChrimson", "Fly2"),
        # os.path.join(load.NAS2_DIR_JB, "221216_DfdxGCaMP6s_tdTom_aDN2xCsChrimson", "Fly1"),
        os.path.join(load.NAS2_DIR_JB, "230105_DfdxGCaMP6s_tdTom_aDN2xCsChrimson", "Fly3"),
    ]
    for fly_dir in fly_dirs:
        trial_dirs = load.get_trials_from_fly(fly_dir, contains="xz", exclude="wheel")[0]
        if len(trial_dirs) >= 4:
            trial_dirs = trial_dirs[:2] + trial_dirs[-2:]
            CO2 = True
        else:
            trial_dirs = trial_dirs[:2]
            CO2 = False
        green_stacks = []
        twop_dfs = []
        for trial_dir in trial_dirs:
            print(trial_dir)
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl")
            twop_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, "twop_df.pkl")
            twop_dfs.append(twop_df)
            green_stacks.append(os.path.join(trial_dir, load.PROCESSED_FOLDER, "green_com_warped.tif"))

            get_beh_info_to_twop_df(beh_df, twop_df, twop_df_out_dir=twop_df)
            add_beh_state_to_twop_df(twop_df, twop_df_out_dir=twop_df)
            
            
        video_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER)
        # """
        get_triggered_avg_videos(green_stacks[:2], twop_dfs[:2], video_dir, stim_dur_s=5, walk_rest=True, pre_stim_baseline_dur_s=1,
                                 # stand="exclude", appendix="std_exclude_stim",
                                 stand="all", appendix="", 
                                 highp=False)  # stand="exclude", appendix="std_exclude_stim"
        
        get_triggered_beh_neural_plot(twop_dfs[:2], plot_name="no CO2", plot_out_dir=video_dir, stim_dur_s=5, pre_stim_baseline_dur_s=1)
        make_behaviour_grid_video(trial_dirs[:2])
        # if CO2:
        #     get_triggered_avg_videos(green_stacks[-2:], twop_dfs[-2:], video_dir, stim_dur_s=5, walk_rest=False, pre_stim_baseline_dur_s=1, appendix="_co2")
        #     make_behaviour_grid_video(trial_dirs[-2:], appendix="_CO2")
        