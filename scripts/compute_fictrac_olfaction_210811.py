import os, sys
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from time import time
from scipy.ndimage import gaussian_filter1d, median_filter

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
OUT_PATH = os.path.join(MODULE_PATH, "outputs")

import utils2p
import utils2p.synchronization
from utils2p.synchronization import get_lines_from_h5_file, process_cam_line, process_stimulus_line, crop_lines, get_times, SyncMetadata

from longterm import load, utils
from longterm import plot as myplt
from longterm.plot.videos import make_all_odour_condition_videos

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

def process_lines(sync_file, sync_metadata_file, seven_camera_metadata_file=None):
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

def get_sync_signals(trial_dir, rep_time=30, stim_time=10, sync_out_file="sync.pkl", overwrite=False, sync_to_fall=True):
    sync_file = utils2p.find_sync_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)
    sync_out_file = os.path.join(trial_dir, sync_out_file)
    if os.path.isfile(sync_out_file) and not overwrite:
        with open(sync_out_file, "rb") as f:
            processed_lines = pickle.load(f)
    else:
        processed_lines = process_lines(sync_file, sync_metadata_file, seven_camera_metadata_file)
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

    return t_cam, condition_signals, list_i_cam_start, list_i_cam_end, list_cond_names

def get_v_th(trial_dir):
    trial_image_dir = os.path.join(trial_dir, "behData", "images")
    fictrac_data_file = glob.glob(os.path.join(trial_image_dir, "camera*.dat"))[0]

    col_names = np.arange(25) + 1
    df = pd.read_csv(fictrac_data_file, header=None, names=col_names)

    v_raw = df[19]
    th_raw = df[18]

    r_ball = 5  # mm

    v = gaussian_filter1d(median_filter(v_raw, size=5), sigma=10) * r_ball  # rad/s == mm/s on ball with 1mm radius
    th = (gaussian_filter1d(median_filter(th_raw, size=5), sigma=10) - np.pi) / np.pi * 180
    return v, th

def plot_olfac_conditions(t, signals, conditions, start_indices, t_stim=10, t_plot=(-5,15), 
                          trial_name="", signal_names=["v (mm/s)", "orientation (°)"], integrate=False):
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

    return fig

def main_preprocess(trial_dirs):
    # file_dir = os.path.join(OUT_PATH, "210812_olfactory_test.pdf")
    # with PdfPages(file_dir) as pdf:
    for trial_dir in tqdm(trial_dirs):
        try:
            # trial_name = trial_dir[13:]
            t_cam, condition_signals, list_i_cam_start, list_i_cam_end, list_cond_names = get_sync_signals(trial_dir)
            out_info = {"t_cam": t_cam,
                        "start_cam_frames": list_i_cam_start,
                        "stop_cam_frames": list_i_cam_end,
                        "condition_list": list_cond_names}
            out_dir = os.path.join(trial_dir, "paradigm.pkl")
            with open(out_dir, "wb") as f:
                pickle.dump(out_info, f)
            """
            v, th = get_v_th(trial_dir)

            fig = plot_olfac_conditions(t=t_cam, signals=(v, th), conditions=list_cond_names,
                                        start_indices=list_i_cam_start, trial_name=trial_name)

            pdf.savefig(fig)
            plt.close(fig)  
            """
        except Exception as e:
            print("error during trial: ", trial_dir)
            print(str(e))

def main_plots(trial_dirs):
    file_dir = os.path.join(OUT_PATH, "210826_olfactory_test.pdf")
    out_infos = []
    vs = []
    ths = []
    with PdfPages(file_dir) as pdf:
        for trial_dir in tqdm(trial_dirs):
            try:
                trial_name = trial_dir[13:]
                out_dir = os.path.join(trial_dir, "paradigm.pkl")
                with open(out_dir, "rb") as f:
                    out_info = pickle.load(f)
                out_infos.append(out_info)

                v, th = get_v_th(trial_dir)
                vs.append(v)
                ths.append(th)

                fig = plot_olfac_conditions(t=out_info["t_cam"], signals=(v, th),
                                            conditions=out_info["condition_list"],
                                            start_indices=out_info["start_cam_frames"],
                                            trial_name=trial_name)

                pdf.savefig(fig)
                plt.close(fig)  
                
            except Exception as e:
                print("error during trial: ", trial_dir)
                print(str(e))

        v = np.concatenate(vs)
        th = np.concatenate(ths)
        conditions = np.concatenate([out_info["condition_list"] for out_info in out_infos])
        start_indices = []
        N_samples = 0
        for out_info in out_infos:
            inds = np.array(out_info["start_cam_frames"]) + N_samples
            N_samples += len(out_info["t_cam"])
            start_indices = np.concatenate((start_indices, inds.astype(int)))

        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=conditions,
                                            start_indices=start_indices.astype(int),
                                            trial_name="all trials")
        pdf.savefig(fig)
        plt.close(fig)

        rest_inds = []
        walk_inds = []
        rest_conds = []
        walk_conds = []
        for ind, cond in zip(start_indices.astype(int), conditions):
            if np.mean(v[ind-200:ind]) < 0.01:
                rest_inds.append(ind)
                rest_conds.append(cond)
            else:
                walk_inds.append(ind)
                walk_conds.append(cond)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_conds,
                                            start_indices=rest_inds,
                                            trial_name="resting before stimulus")
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_conds,
                                            start_indices=walk_inds,
                                            trial_name="walking before stimulus")
        pdf.savefig(fig)
        plt.close(fig)

        # plot integrated signals
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=conditions,
                                            start_indices=start_indices.astype(int),
                                            trial_name="all trials integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
        for ind, cond in zip(start_indices.astype(int), conditions):
            if np.mean(v[ind-200:ind]) < 0.01:
                rest_inds.append(ind)
                rest_conds.append(cond)
            else:
                walk_inds.append(ind)
                walk_conds.append(cond)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_conds,
                                            start_indices=rest_inds,
                                            trial_name="resting before stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_conds,
                                            start_indices=walk_inds,
                                            trial_name="walking before stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
          
def main_videos(trial_dirs):
    video_dirs = [os.path.join(trial_dir, "behData", "images", "camera_5.mp4") for trial_dir in trial_dirs]
    paradigm_dirs = [os.path.join(trial_dir, "paradigm.pkl") for trial_dir in trial_dirs]
    out_dir = OUT_PATH
    video_name = "210826_olfac_video"
    make_all_odour_condition_videos(video_dirs, paradigm_dirs, out_dir, video_name, 
                                    frame_range=[-500,1500], stim_length=1000, frame_rate=None,
                                    size=(120,-1))


if __name__ == "__main__":
    fly_dirs = [# os.path.join(load.NAS2_DIR_JB, "210727_PR_olfac_test", "Fly3"),
                # os.path.join(load.NAS2_DIR_JB, "210727_PR_olfac_test", "Fly4"),
                # os.path.join(load.NAS2_DIR_JB, "210728_PR_olfac_test", "Fly1"),
                # os.path.join(load.NAS2_DIR_JB, "210728_PR_olfac_test", "Fly2"),
                # os.path.join(load.NAS2_DIR_JB, "210729_PR_olfac_test", "Fly2"),
                os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly1"),
                os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly2"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly002"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly003"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly004"),
                ]
    trial_dirs = []
    for fly_dir in fly_dirs:
        for d in os.listdir(fly_dir):
            if os.path.isdir(os.path.join(fly_dir,d)) and d != "processed":
                trial_dirs.append(os.path.join(fly_dir,d))

    trial_dirs_40 = [trial_dir for trial_dir in trial_dirs if "40ML" in trial_dir]
    trial_dirs_100 = [trial_dir for trial_dir in trial_dirs if "100ML" in trial_dir]
    trial_dirs_10 = [trial_dir for trial_dir in trial_dirs if "10ML" in trial_dir]
    trial_dirs = trial_dirs_40
    # trial_dirs.pop(5)
    # trial_dirs.pop(3)

    print(trial_dirs)
    # main_preprocess(trial_dirs)
    print("making plots")
    # main_plots(trial_dirs)
    print("making videos")
    main_videos(trial_dirs)
    #TODO: optimise grid_size function in utils_video --> for 9 videos, returns 4x3
