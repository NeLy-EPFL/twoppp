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
from longterm.behaviour.fictrac import get_fictrac_df
from longterm.behaviour.olfaction import get_sync_signals_olfaction, conditions, plot_olfac_conditions

def main_preprocess(trial_dirs):
    # file_dir = os.path.join(OUT_PATH, "210812_olfactory_test.pdf")
    # with PdfPages(file_dir) as pdf:
    for trial_dir in tqdm(trial_dirs):
        try:
            # trial_name = trial_dir[13:]
            t_cam, condition_signals, list_i_cam_start, list_i_cam_end, list_cond_names = get_sync_signals_olfaction(trial_dir)
            """
            df = get_fictrac_df(trial_dir)
            v = df["v"]
            th = df["th"]
            fig = plot_olfac_conditions(t=t_cam, signals=(v, th), conditions=list_cond_names,
                                        start_indices=list_i_cam_start, trial_name=trial_name)

            pdf.savefig(fig)
            plt.close(fig)  
            """
        except Exception as e:
            print("error during trial: ", trial_dir)
            print(str(e))

def main_plots(trial_dirs, file_name):
    file_dir = os.path.join(OUT_PATH, file_name)
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

                df = get_fictrac_df(trial_dir)
                v = df["v"].to_numpy()
                th = df["th"].to_numpy()
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
            if np.mean(v[ind-200:ind]) < 1:  # mm/s
                rest_inds.append(ind)
                rest_conds.append(cond)
            else:
                walk_inds.append(ind)
                walk_conds.append(cond)

        rest_during_inds = []
        walk_during_inds = []
        rest_during_conds = []
        walk_during_conds = []
        for ind, cond in zip(start_indices.astype(int), conditions):
            if np.mean(v[ind+500:ind+900]) < 1:  # mm/s
                rest_during_inds.append(ind)
                rest_during_conds.append(cond)
            else:
                walk_during_inds.append(ind)
                walk_during_conds.append(cond)

        rest_middle_inds = []
        walk_middle_inds = []
        rest_middle_conds = []
        walk_middle_conds = []
        for ind, cond in zip(start_indices.astype(int), conditions):
            if np.mean(v[ind+250:ind+500]) < 1:  # mm/s
                rest_middle_inds.append(ind)
                rest_middle_conds.append(cond)
            else:
                walk_middle_inds.append(ind)
                walk_middle_conds.append(cond)

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

        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_during_conds,
                                            start_indices=rest_during_inds,
                                            trial_name="resting/grooming during end of stimulus")
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_during_conds,
                                            start_indices=walk_during_inds,
                                            trial_name="walking during end of stimulus")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_middle_conds,
                                            start_indices=rest_middle_inds,
                                            trial_name="resting/grooming during middle of stimulus")
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_middle_conds,
                                            start_indices=walk_middle_inds,
                                            trial_name="walking during middle of stimulus")
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
            if np.mean(v[ind-200:ind]) < 1:  # mm/s
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

        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_during_conds,
                                            start_indices=rest_during_inds,
                                            trial_name="resting/grooming during end of stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_during_conds,
                                            start_indices=walk_during_inds,
                                            trial_name="walking during end of stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=rest_middle_conds,
                                            start_indices=rest_middle_inds,
                                            trial_name="resting/grooming during middle of stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_olfac_conditions(t=out_infos[0]["t_cam"], signals=(v, th),
                                            conditions=walk_middle_conds,
                                            start_indices=walk_middle_inds,
                                            trial_name="walking during middle of stimulus integrated", integrate=True,
                                            signal_names=["distance (mm)", "integrated heading (°)"])
        pdf.savefig(fig)
        plt.close(fig)
          
def main_videos(trial_dirs, video_name):
    video_dirs = [os.path.join(trial_dir, "behData", "images", "camera_5.mp4") for trial_dir in trial_dirs]
    paradigm_dirs = [os.path.join(trial_dir, "paradigm.pkl") for trial_dir in trial_dirs]
    out_dir = OUT_PATH
    make_all_odour_condition_videos(video_dirs, paradigm_dirs, out_dir, video_name, 
                                    frame_range=[-500,1500], stim_length=1000, frame_rate=None,
                                    size=(120,-1))


if __name__ == "__main__":
    fly_dirs = [# os.path.join(load.NAS2_DIR_JB, "210727_PR_olfac_test", "Fly3"),
                # os.path.join(load.NAS2_DIR_JB, "210727_PR_olfac_test", "Fly4"),
                # os.path.join(load.NAS2_DIR_JB, "210728_PR_olfac_test", "Fly1"),
                # os.path.join(load.NAS2_DIR_JB, "210728_PR_olfac_test", "Fly2"),
                # os.path.join(load.NAS2_DIR_JB, "210729_PR_olfac_test", "Fly2"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly1"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly2"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly002"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly003"),
                # os.path.join(load.NAS2_DIR_JB, "210826_PR_olfac_test", "Fly004"),
                os.path.join(load.NAS2_DIR_JB, "210908_PR_olfac", "Fly1"),
                ]
    trial_dirs = load.get_trials_from_fly(fly_dirs, startswith="0")
    trial_dirs = [trial_dir for fly_trial_dirs in trial_dirs for trial_dir in fly_trial_dirs]

    trial_dirs_40 = [trial_dir for trial_dir in trial_dirs if "40ML" in trial_dir]
    trial_dirs_100 = [trial_dir for trial_dir in trial_dirs if "100ML" in trial_dir]
    trial_dirs_10 = [trial_dir for trial_dir in trial_dirs if "10ML" in trial_dir]
    trial_dirs = trial_dirs_40
    # trial_dirs.pop(5)
    # trial_dirs.pop(3)

    print(trial_dirs)
    # main_preprocess(trial_dirs)
    print("making plots")
    # main_plots(trial_dirs, "210908_olfactory.pdf")
    print("making videos")
    main_videos(trial_dirs, video_name="210908_olfac_video")
    #TODO: optimise grid_size function in utils_video --> for 9 videos, returns 4x3
