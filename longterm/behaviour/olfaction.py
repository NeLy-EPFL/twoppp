"""
sub-module to analyse data from recordings with olfactory stimulation
"""
import os
import sys
import pickle
import glob
from scipy.ndimage import gaussian_filter1d, median_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import utils2p
import utils2p.synchronization
from utils2p.synchronization import get_lines_from_h5_file, process_cam_line, process_stimulus_line, crop_lines, get_times, SyncMetadata

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)
OUTPUT_PATH = os.path.join(MODULE_PATH, "outputs")

from longterm import load, utils
from longterm import plot as myplt


conditions_old = ["None", "Odor1", "Odor2", "Odor3", "Odor4", "Odor5", "Odor6",
                    "Odor1R", "Odor2R",   # 7,8
                    "Odor1L", "Odor2L",   # 9,10
                    "Odor1B", "Odor2B", "WaterB"]   # 11,12,13
# since 26.08.2021
conditions = ["None", "Odor1", "Odor2", "Odor3", "Odor4", "Odor5", "Odor6",
                    "Odor1R", "Odor2R",   # 7,8
                    "Odor1L", "Odor2L",   # 9,10
                    "Odor1B", "Odor2B", "WaterB",   # 11,12,13
                    "Florian1", "Florian2", "Florian3"]  # 14,15

def process_lines_olfaction(sync_file, sync_metadata_file, seven_camera_metadata_file=None):
    """
    This function extracts all the standard lines and processes them.
    It works for both microscopes.
    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    sync_metadata_file : str
        Path to the synchronization metadata file.
    metadata_2p_file : str
        Path to the ThorImage metadata file.
    seven_camera_metadata_file : str
        Path to the metadata file of the 7 camera system.
    Returns
    -------
    processed_lines : dictionary
        Dictionary with all processed lines.
    """
    processed_lines = {}
    try:
        processed_lines["odor"], processed_lines["pid"], processed_lines["Cameras"] = get_lines_from_h5_file(sync_file, ["odor", "pid", "Cameras",])
        processed_lines["Cameras"] = process_cam_line(processed_lines["Cameras"], seven_camera_metadata_file)
    except:
        processed_lines["odor"], processed_lines["Cameras"] = get_lines_from_h5_file(sync_file, ["odor", "Cameras",])
    
    # Get times of ThorSync ticks
    metadata = SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = get_times(len(processed_lines["odor"]), freq)
    processed_lines["Times"] = times

    return processed_lines

def get_sync_signals_olfaction(trial_dir, rep_time=30, stim_time=10, sync_out_file="sync.pkl", paradigm_out_file="paradigm.pkl", 
                               overwrite=False, sync_to_fall=False, index_df=None, df_out_dir=None):
    sync_file = utils2p.find_sync_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)
    sync_out_file = os.path.join(trial_dir, sync_out_file)
    paradigm_out_file = os.path.join(trial_dir, paradigm_out_file)

    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    elif index_df is not None:
        assert isinstance (index_df, pd.DataFrame)

    data_read = False
    
    if os.path.isfile(paradigm_out_file) and not overwrite:
        try:
            with open(paradigm_out_file, "rb") as f:
                out_info = pickle.load(f)
            t_cam = out_info["t_cam"]
            condition_signals = out_info["condition_signals"]
            list_i_cam_start = out_info["start_cam_frames"]
            list_i_cam_end = out_info["stop_cam_frames"]
            list_cond_names = out_info["condition_list"]
            data_read = True
        except:
            print("Error while reading paradigm from file. Will re-compute.")

    if not data_read:
        if os.path.isfile(sync_out_file) and not overwrite:
            with open(sync_out_file, "rb") as f:
                processed_lines = pickle.load(f)
        else:
            processed_lines = process_lines_olfaction(sync_file, sync_metadata_file, seven_camera_metadata_file)
            with open(sync_out_file, "wb") as f:
                pickle.dump(processed_lines, f)

        processed_lines_crop = {}
        for k in processed_lines.keys():
            processed_lines_crop[k] = processed_lines[k][processed_lines["Cameras"]>=0]

        t = processed_lines_crop["Times"]
        t = t - t[0]
        cam = processed_lines_crop["Cameras"]
        odor = gaussian_filter1d(processed_lines_crop["odor"], sigma=100)
        t_cam = np.hstack((0, t[np.where(np.diff(cam))[0]]))

        min_pwm = 0  # np.quantile(processed_lines_crop["odor"], 0.001)
        max_pwm = np.quantile(processed_lines_crop["odor"], 0.99)   

        odor_steps = len(conditions)
        odor_quant = (np.round((odor-min_pwm) / (max_pwm-min_pwm) * odor_steps)).astype(int)

        f_s_sync = 1 / np.mean(np.diff(t))
        f_s_cam = np.mean(np.diff(cam)) / np.mean(np.diff(t))

        if not sync_to_fall:
            odor_thres = processed_lines_crop["odor"] > 1
            i = 0
            list_i_start = []
            while i < len(odor_thres)-1:
                i += 1
                if odor_thres[i]:
                    list_i_start.append(i)
                    i += int(f_s_sync * (rep_time - 1))
        else:
            odor_thres = 3.4
            switch = np.logical_and(odor <= odor_thres, np.roll(odor, shift=1) > odor_thres)
            list_i_start = np.where(switch)[0][:-1]

        list_t_start = t[list_i_start]
        list_i_cam_start = cam[list_i_start]

        list_i_end = [start + int(f_s_sync)*stim_time for start in list_i_start]
        list_t_end = list_t_start + stim_time
        list_i_cam_end = list_i_cam_start + int(f_s_cam)*stim_time

        list_cond = [odor_quant[int(np.mean([start, end]))] for start, end in zip(list_i_start, list_i_end)]
        list_cond_names = [conditions[cond] for cond in list_cond]

        condition_signals = []
        for c in np.unique(list_cond_names):
            tmp = np.zeros_like(t_cam).astype(bool)
            for cond, i_start, i_end in zip(list_cond_names, list_i_cam_start, list_i_cam_end):
                if cond == c:
                    tmp[i_start:i_end] = True
            condition_signals.append(tmp)

        out_info = {"t_cam": t_cam,
                    "condition_signals": condition_signals,
                    "start_cam_frames": list_i_cam_start,
                    "stop_cam_frames": list_i_cam_end,
                    "condition_list": list_cond_names}
        with open(paradigm_out_file, "wb") as f:
            pickle.dump(out_info, f)

    if index_df is not None:
        assert len(index_df) == len(t_cam)
        binary_cond_signal = np.zeros_like(t_cam).astype(bool)
        condition_name_signal = [""] * len(t_cam)
        for i_start, i_end, cond in zip(list_i_cam_start, list_i_cam_end, list_cond_names):
            binary_cond_signal[i_start:i_end] = True
            for i in range(i_start, i_end):
                condition_name_signal[i] = cond
        index_df["olfac_stim"] = binary_cond_signal
        index_df["olfac_cond"] = condition_name_signal

    if df_out_dir is not None:
        index_df.to_pickle(df_out_dir)

    return t_cam, condition_signals, list_i_cam_start, list_i_cam_end, list_cond_names

def plot_olfac_conditions(t, signals, conditions, start_indices, t_stim=10, t_plot=(-5,15),
                          trial_name="", signal_names=["v (mm/s)", "orientation (°)"], integrate=False,
                          return_signals=False):
    # t = np.arange(len(signals[0])) / 100  # TODO: fix this
    f_s = 100  # int(1 / np.mean(np.diff(t)))
    i_plot = (t_plot[0]*f_s, t_plot[1]*f_s+1)
    t_vec = np.arange(t_plot[0], t_plot[1]+1/f_s, 1/f_s)
    cond_signal = np.hstack((np.zeros((np.abs(t_plot[0]*f_s))), 
                             np.ones((t_stim*f_s)), 
                             np.zeros(((t_plot[-1]-t_stim)*f_s))
                             ))

    N_cond = len(np.unique(conditions))
    N_signals = len(signals)

    signals_to_avg = [[[] for c in range(N_cond)]
                          for s in range(N_signals)]

    fig, axs = plt.subplots(N_signals, N_cond, figsize=(1+4.5*N_cond, 2+3*N_cond), sharex=True, sharey="row", squeeze=False)

    for i_c, c in enumerate(np.unique(conditions)):
        if i_c == 0:
            for i_s in range(N_signals):
                myplt.shade_walk_rest(cond_signal, np.zeros_like(cond_signal), x=t_vec, ax=axs[i_s,i_c])
                axs[i_s,i_c].set_ylabel(signal_names[i_s])
        else:
            for i_s in range(N_signals):
                myplt.shade_walk_rest(np.zeros_like(cond_signal), cond_signal, x=t_vec, ax=axs[i_s,i_c])
        axs[-1,i_c].set_xlabel("t (s)")
        axs[0,i_c].set_title(c)

        for i_s, (signal, name) in enumerate(zip(signals, signal_names)):
            for cond, i_start in zip(conditions, start_indices):
                if cond == c:
                    s_tmp = signal[i_start+i_plot[0]:i_start+i_plot[1]]
                    if len(s_tmp) != len(t_vec):
                        continue
                    if integrate:
                        stim_signal = s_tmp[np.abs(t_plot[0]*f_s):]  # np.abs(t_plot[0]*f_s)+t_stim*f_s]
                        if "/s" in name:
                            stim_signal = (np.cumsum(stim_signal) - stim_signal[0]) / f_s
                        else:
                            stim_signal = (np.cumsum(stim_signal) - stim_signal[0]) / f_s #TODO: remove this
                        integrated_signal = np.hstack((np.zeros((np.abs(t_plot[0]*f_s))),
                             stim_signal,
                             # np.zeros(((t_plot[-1]-t_stim)*f_s))
                             ))
                        s_tmp = integrated_signal
                    signals_to_avg[i_s][i_c].append(s_tmp)

                    axs[i_s, i_c].plot(t_vec, s_tmp, "k", alpha=0.1)

            myplt.plot_mu_sem(mu=np.mean(signals_to_avg[i_s][i_c], axis=0),
                              err=utils.conf_int(signals_to_avg[i_s][i_c], axis=0),
                              x=t_vec, ax=axs[i_s,i_c], color="k")

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(trial_name)
    fig.tight_layout()
    if return_signals:
        return fig, signals_to_avg
    return fig


def average_neural_across_repetitions(beh_dfs, to_average, output_dir=None, condition="WaterB", t_range=[-5, 15], twop_fs=16):
    N_rep = 0
    
    for i_t, (beh_df, stack) in enumerate(tqdm(zip(beh_dfs, to_average))):
        if not isinstance(beh_df, pd.DataFrame):
            beh_df = pd.read_pickle(beh_df)
        stack = utils.get_stack(stack)
        N_frames = len(stack)
        if i_t == 0:
            output = np.zeros((1+int(t_range[1]*twop_fs)-int(t_range[0]*twop_fs), stack.shape[1], stack.shape[2]))
        stim_start = np.argwhere(np.diff(beh_df["olfac_stim"].to_numpy().astype(int))==1).flatten()
        stim_start = [s for s in stim_start if beh_df["olfac_cond"].to_numpy()[s+2] == condition]
        twop_stim_start = beh_df["twop_index"].to_numpy()[stim_start]
        

        for this_stim_start in twop_stim_start:
            start = this_stim_start + int(t_range[0]*twop_fs)
            stop = this_stim_start + int(t_range[1]*twop_fs) + 1
            if stop < N_frames and start > 0:
                output += stack[start:stop,:,:]
                N_rep += 1


    output /= N_rep
    if output_dir is not None:
        utils.save_stack(output_dir, output)
    return output



