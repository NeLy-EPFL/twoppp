import os, sys
import pickle
import numpy as np
import pandas as pd
import gc

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

import utils2p.synchronization

from twoppp import load
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes, reduce_during_2p_frame
from twoppp.behaviour.optic_flow import get_opflow_df, resting, forward_walking
from twoppp.utils import readlines_tolist, get_stack
from twoppp.rois import get_roi_signals_df
from twoppp.plot import videos
fly_dirs = [
    "/mnt/NAS2/LH/210512/fly3",
    "/mnt/NAS2/LH/210519/fly1",
    "/mnt/NAS2/LH/210521/fly1",
    "/mnt/NAS2/LH/210526/fly2",
    "/mnt/NAS2/LH/210527/fly4",

    "/mnt/NAS2/LH/210602/fly2",
    "/mnt/NAS2/LH/210603/fly2"# ,
    # "/mnt/NAS2/LH/210604/fly3"
    ]

all_selected_trials = [[2,5,11],  # 0,2,3,4,5,8,11],  # 
                           [3,8,12],
                           [3,7,12],
                           [2,7,11],
                           [3,6,10],

                           [3,6,10],
                           [1,6,10]# ,
                           # [3,5,7,10]
                           ]

all_thres_rest = [0.01, 0.015, 0.01, 0.01, 0.015, \
                  0.015, 0.015]
all_thres_walk = [0.035, 0.03, 0.025, 0.02, 0.03, \
                  0.04, 0.04]

for i_fly, (fly_dir, selected_trials, thres_walk, thres_rest) in enumerate(zip(fly_dirs, all_selected_trials, all_thres_walk, all_thres_rest)):
    if i_fly < 4:
        continue
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
    for i_trial in range(len(trial_dirs)):
        _, trial_name = os.path.split(trial_dirs[i_trial])
        trial_names.append(trial_name)
        processed_dir = os.path.join(trial_dirs[i_trial], load.PROCESSED_FOLDER)
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
        opflow_out_dir = os.path.join(processed_dir, "opflow_df.pkl")
        df3d_out_dir = os.path.join(processed_dir, "beh_df.pkl")
        twop_out_dir = os.path.join(processed_dir, "twop_df.pkl")
        # load opflow
        opflow_df = pd.read_pickle(opflow_out_dir)
        twop_df = pd.read_pickle(twop_out_dir)

        twop_index = opflow_df["twop_index"]
        twop_df["velForw"] = reduce_during_2p_frame(twop_index, opflow_df["velForw"])[:3840]
        twop_df["velSide"] = reduce_during_2p_frame(twop_index, opflow_df["velSide"])[:3840]
        twop_df["velTurn"] = reduce_during_2p_frame(twop_index, opflow_df["velTurn"])[:3840]

        twop_df = resting(twop_df, thres_rest=thres_rest, winsize=16)
        twop_df = forward_walking(twop_df, thres_walk=thres_walk, winsize=4)

        rest = np.argwhere(twop_df["rest"].to_numpy()).flatten()
        rests.append(rest)
        walk = np.argwhere(twop_df["walk"].to_numpy()).flatten()
        walks.append(walk)

        """
        beh_df = pd.read_pickle(df3d_out_dir)
        # downsample opflow to behavioural frames
        t_beh = beh_df["t"].to_numpy()
        t_opflow = opflow_df["t"].to_numpy()

        beh_in_opflow_index = np.zeros_like(t_opflow).astype(np.int64)
        i_t_beh = 0
        for i_t, t in enumerate(t_opflow):
            if t < t_beh[i_t_beh]:
                beh_in_opflow_index[i_t] = i_t_beh
            else:
                i_t_beh += 1
                if i_t_beh == len(t_beh):
                    beh_in_opflow_index[i_t:] = -9223372036854775808  # smallest possible uint64 number
                    break
                beh_in_opflow_index[i_t] = i_t_beh

        # reduce during behaviour frame
        beh_df["velForw"] = reduce_during_2p_frame(beh_in_opflow_index, opflow_df["velForw"])
        beh_df["velSide"] = reduce_during_2p_frame(beh_in_opflow_index, opflow_df["velSide"])
        beh_df["velTurn"] = reduce_during_2p_frame(beh_in_opflow_index, opflow_df["velTurn"])

        # compute walking and resting
        beh_df = resting(beh_df, thres_rest=thres_rest, winsize=100)
        beh_df = forward_walking(beh_df, thres_walk=thres_walk, winsize=25)
        
        rest = np.argwhere(beh_df["rest"].to_numpy()).flatten()
        rests.append(rest)
        walk = np.argwhere(beh_df["walk"].to_numpy()).flatten()
        walks.append(walk)
        """

    # define all file names
    dffs = [os.path.join(trial_dir, "processed", "dff_denoised_t1.tif") for trial_dir in trial_dirs]
    green_dirs = [os.path.join(trial_dir, "processed", "green_com_warped.tif") for trial_dir in trial_dirs]
    red_dirs = [os.path.join(trial_dir, "processed", "red_com_warped.tif") for trial_dir in trial_dirs]
    fly_processed_dir = os.path.join(fly_dir, "processed")
    text = trial_names

    try:
        mask = get_stack(os.path.join(fly_processed_dir, "dff_mask_denoised_t1.tif")) > 0
    except:
        mask = None
    if i_fly != 0:
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
                                        share_lim=True, 
                                        share_mask=True,
                                        blur=0, mask=mask, crop=None,
                                        text=text,
                                        downsample=None,
                                        select_frames=walks)
    
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
                                    share_lim=True, 
                                    share_mask=True,
                                    blur=0, mask=mask, crop=None,
                                    text=text,
                                    downsample=None,
                                    select_frames=rests)

    gc.collect()