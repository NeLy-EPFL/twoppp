import os, sys
import pickle
import numpy as np
import pandas as pd
import gc
from shutil import copyfile

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

import utils2p.synchronization

from longterm import load
from longterm import fly_dirs, all_selected_trials, conditions
from longterm.behaviour.synchronisation import get_synchronised_trial_dataframes, reduce_during_2p_frame, reduce_min, reduce_bin_75p
from longterm.behaviour.optic_flow import get_opflow_df, resting, forward_walking
from longterm.utils import readlines_tolist, get_stack
from longterm.rois import get_roi_signals_df
from longterm.plot import videos

OVERWRITE_DF = False
TARGET_DIR = os.path.join(load.NAS2_DIR_JB, "longterm", "behaviour_video_snippets")

def main():
    fly_dirs_sucr = [fly_dir for i_fly, fly_dir in enumerate(fly_dirs) 
                     if "sucr" in conditions[i_fly]]
    all_selected_trials_sucr = [selected_trials for i_fly, selected_trials in enumerate(all_selected_trials)
                                if "sucr" in conditions[i_fly]]

    for i_fly, (fly_dir, selected_trials, condition) in enumerate(zip(fly_dirs, all_selected_trials, conditions)):
        if i_fly < 1:
            continue
        thres_walk = 0.03
        thres_rest = 0.01
        print("====="+fly_dir)
        tmp, fly = os.path.split(fly_dir)
        fly = int(fly[-1:])
        _, date = os.path.split(tmp)
        date = int(date)
        trial_dirs = readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
        trial_dirs = [trial_dirs[i] for i in selected_trials]
        sync_trial_dirs = readlines_tolist(os.path.join(fly_dir, "sync_trial_dirs.txt"))
        sync_trial_dirs = [sync_trial_dirs[i] for i in selected_trials]
        beh_trial_dirs = readlines_tolist(os.path.join(fly_dir, "beh_trial_dirs.txt"))
        beh_trial_dirs = [beh_trial_dirs[i] for i in selected_trials]

        trial_names = []

        rests = []
        walks = []
        print("===== creating walk/rest dataframes")
        for i_trial in range(len(trial_dirs)):
            _, trial_name = os.path.split(trial_dirs[i_trial])
            trial_names.append(trial_name)
            processed_dir = os.path.join(trial_dirs[i_trial], load.PROCESSED_FOLDER)
            if not os.path.isdir(processed_dir):
                os.makedirs(processed_dir)
            opflow_out_dir = os.path.join(processed_dir, "opflow_df.pkl")
            # df3d_out_dir = os.path.join(processed_dir, "beh_df.pkl")
            twop_out_dir = os.path.join(processed_dir, "twop_df.pkl")
            # load opflow
            twop_df = pd.read_pickle(twop_out_dir)
            if OVERWRITE_DF or not all([key in twop_df.keys() for key in ["walk", "rest"]]):
                opflow_df = pd.read_pickle(opflow_out_dir)

                twop_df = twop_df[:3840]
                twop_index = opflow_df["twop_index"]
                twop_df["velForw"] = reduce_during_2p_frame(twop_index, opflow_df["velForw"])[:3840]
                twop_df["velSide"] = reduce_during_2p_frame(twop_index, opflow_df["velSide"])[:3840]
                twop_df["velTurn"] = reduce_during_2p_frame(twop_index, opflow_df["velTurn"])[:3840]
                twop_df["walk_resamp"] = reduce_during_2p_frame(twop_index, opflow_df["walk"], function=reduce_min)[:3840]  # reduce_bin_75p
                twop_df["rest_resamp"] = reduce_during_2p_frame(twop_index, opflow_df["rest"], function=reduce_min)[:3840]  # reduce_bin_75p


                twop_df = resting(twop_df, thres_rest=thres_rest, winsize=16)
                twop_df = forward_walking(twop_df, thres_walk=thres_walk, winsize=4)

                twop_df.to_pickle(twop_out_dir)

            rest = np.argwhere(twop_df["rest"].to_numpy()).flatten()
            rests.append(rest)
            walk = np.argwhere(twop_df["walk"].to_numpy()).flatten()
            walks.append(walk)


        if "caff" in condition:
            dff_video_share_lim = False
            dff_video_log_lim = False  # [False, False, False, True]
            last_trial = np.where(["007" in trial for trial in trial_dirs])[0]
            if not last_trial.size:
                last_trial = np.where(["006" in trial for trial in trial_dirs])[0]
            if not last_trial.size:
                last_trial = [-1]
            last_trial = last_trial[0]
            i_trials = [0,2,3,last_trial]
            if "210723 fly 2" in condition:
                i_trials = [1,2,3,last_trial]
        else:
            dff_video_share_lim = True
            dff_video_log_lim = False
            i_trials = [0,1,2,len(trial_dirs)-1]
        
        dffs = [os.path.join(trial_dir, "processed", "dff_denoised_t1_corr.tif")
            for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in i_trials]  # dff_denoised_t1.tif
        green_dirs = [os.path.join(trial_dir, "processed", "green_com_warped.tif")
            for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in i_trials]
        red_dirs = [os.path.join(trial_dir, "processed", "red_com_warped.tif")
            for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in i_trials]
        trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(trial_dirs)
            if i_trial in i_trials]
        beh_trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(beh_trial_dirs)
            if i_trial in i_trials]
        sync_trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(sync_trial_dirs)
            if i_trial in i_trials]
        fly_processed_dir = os.path.join(fly_dir, "processed")
        text = [trial_name for i_trial, trial_name in enumerate(trial_names)
            if i_trial in i_trials]
        walks = [walk for i_trial, walk in enumerate(walks)
            if i_trial in i_trials]
        rests = [rest for i_trial, rest in enumerate(rests)
            if i_trial in i_trials]

        try:
            mask = get_stack(os.path.join(fly_processed_dir, "dff_mask_raw.tif")) > 0  # dff_mask_denoised_t1.tif
        except:
            mask = None
        print("===== making videos")
        if all([len(walk)==0 for walk in walks]):
            print("No walking detected. Omitting video.")
        else:
            videos.make_multiple_video_raw_dff_beh(dffs=dffs,
                                            trial_dirs=trial_dirs,
                                            out_dir=fly_processed_dir,
                                            video_name="walking_snippets",
                                            beh_dirs=beh_trial_dirs,
                                            sync_dirs=sync_trial_dirs,
                                            camera=6,
                                            stack_axes=[0, 1],
                                            greens=green_dirs,
                                            reds=red_dirs,
                                            vmin=0,
                                            vmax=None,
                                            pmin=None,
                                            pmax=99,
                                            share_lim=dff_video_share_lim,
                                            log_lim=dff_video_log_lim,
                                            share_mask=True,
                                            blur=0, mask=mask, crop=None,
                                            text=text,
                                            downsample=None,
                                            select_frames=walks)
            dir_name = os.path.join(TARGET_DIR, condition.replace(" ", "_")+"_walking_snippets.mp4")
            print("Copying to ", dir_name)
            copyfile(os.path.join(fly_processed_dir, "walking_snippets.mp4"), dir_name)
            gc.collect()

        videos.make_multiple_video_raw_dff_beh(dffs=dffs,
                                        trial_dirs=trial_dirs,
                                        out_dir=fly_processed_dir,
                                        video_name="resting_snippets",
                                        beh_dirs=beh_trial_dirs,
                                        sync_dirs=sync_trial_dirs,
                                        camera=6,
                                        stack_axes=[0, 1],
                                        greens=green_dirs,
                                        reds=red_dirs,
                                        vmin=0,
                                        vmax=None,
                                        pmin=None,
                                        pmax=99,
                                        share_lim=dff_video_share_lim,
                                        log_lim=dff_video_log_lim,
                                        share_mask=True,
                                        blur=0, mask=mask, crop=None,
                                        text=text,
                                        downsample=None,
                                        select_frames=rests)
        dir_name = os.path.join(TARGET_DIR, condition.replace(" ", "_")+"_resting_snippets.mp4")
        print("Copying to ", dir_name)
        copyfile(os.path.join(fly_processed_dir, "resting_snippets.mp4"), dir_name)

        gc.collect()


if __name__ == "__main__":
    main()

