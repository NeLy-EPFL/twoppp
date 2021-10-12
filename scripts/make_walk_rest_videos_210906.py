import os, sys
import pickle
import numpy as np
import pandas as pd
import gc
from shutil import copyfile
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
OUT_PATH = os.path.join(MODULE_PATH, "outputs")

import utils2p.synchronization

from longterm import load
from longterm import fly_dirs, all_selected_trials, conditions, all_selected_trials_old
from longterm.behaviour.synchronisation import get_synchronised_trial_dataframes, reduce_during_2p_frame, reduce_min, reduce_bin_75p
from longterm.behaviour.optic_flow import get_opflow_df, resting, forward_walking, clean_rest
from longterm.utils import readlines_tolist, get_stack
from longterm.utils.df import get_norm_dfs
from longterm import rois
from longterm.plot import videos

def make_roi_dff_video_data(roi_dff_signals, roi_mask):
    N_y, N_x = roi_mask.shape
    N_frames, N_rois = roi_dff_signals.shape
    assert len(np.unique(roi_mask)) - 1 == N_rois

    dff_video = np.zeros((N_frames, N_y, N_x))
    for i_roi, roi in enumerate(tqdm(np.unique(roi_mask)[1:])):
        for i_frame in np.arange(N_frames):
            dff_video[i_frame, roi_mask==roi] = 100*roi_dff_signals[i_frame, i_roi]

    return dff_video

def make_dff_maps(dff_dirs, walks, rests, perc=0.99, map_dict_out_dir=None, trial_names=None):
    maxs = []
    mins = []
    means_walk = []
    means_rest = []
    for dff_dir, walk, rest in tqdm(zip(dff_dirs, walks, rests)):
        dff = get_stack(dff_dir)
        maxs.append(np.quantile(dff, (1+perc)/2, axis=0))
        mins.append(np.quantile(dff, (1-perc)/2, axis=0))
        if len(walk) == 0 or np.sum(walk) == 0:
            means_walk.append([])
        else:
            means_walk.append(np.mean(dff[walk,:,:], axis=0))
        if len(rest) == 0 or np.sum(rest) == 0:
            means_rest.append([])
        else:
            means_rest.append(np.mean(dff[rest,:,:], axis=0))
        del dff
    image_min = np.mean(mins, axis=0)
    image_max = np.mean(maxs, axis=0)
    means_walk_norm = []
    means_rest_norm = []
    for mean_walk, mean_rest in zip(means_walk, means_rest):
        if len(mean_walk) == 0:
            means_walk_norm.append([])
        else:
            means_walk_norm.append((mean_walk - image_min) / (image_max - image_min))
        if len(mean_rest) == 0:
            means_rest_norm.append([])
        else:
            means_rest_norm.append((mean_rest - image_min) / (image_max - image_min))
    map_dict = {
        "dff_min": image_min,
        "dff_max": image_max,
        "means_walk": means_walk,
        "means_rest": means_rest,
        "means_walk_norm": means_walk_norm,
        "means_rest_norm": means_rest_norm,
        "trial_names": trial_names
    }
    if map_dict_out_dir is not None:
        with open(map_dict_out_dir, "wb") as f:
            pickle.dump(map_dict, f)
    return map_dict

def plot_dff_maps(map_dict, fly_dir):
    N_trials = len(map_dict["means_rest"])
    fig, axs = plt.subplots(N_trials, 4, figsize=(16,N_trials*3))
    clim = [0, np.maximum(1,np.mean(map_dict["dff_max"]))]  # np.mean(map_dict["dff_min"])
    ims = ["means_walk", "means_rest", "means_walk_norm", "means_rest_norm"]
    clims = [clim, clim, [0,1], [0,1]]
    titles = ["mean: walk", "mean: rest", "normalised mean: walk", "normalised mean: rest"]
    cmaps = [plt.cm.get_cmap("jet"),plt.cm.get_cmap("jet"), plt.cm.get_cmap("viridis"), plt.cm.get_cmap("viridis")]
    for i_trial, axs_row in enumerate(axs):
        for i_col, ax in enumerate(axs_row):
            im = map_dict[ims[i_col]][i_trial]
            if len(im):
                ax.imshow(im, clim=clims[i_col], cmap=cmaps[i_col])
            else:
                ax.axis("off")
            title = titles[i_col]
            if i_col == 0:
                title = map_dict["trial_names"][i_trial] + " " + title
            ax.set_title(title)
    fig.suptitle(fly_dir)
    fig.tight_layout()
    return fig

OVERWRITE_DF = True
TARGET_DIR = os.path.join(load.NAS2_DIR_JB, "longterm", "behaviour_video_snippets")
OVERWRITE_DF_VIDEO = False
MAKE_MAPS = False
MAPS_ONLY = False

def main():
    # fly_dirs_sucr = [fly_dir for i_fly, fly_dir in enumerate(fly_dirs) 
    #                  if "sucr" in conditions[i_fly]]
    # all_selected_trials_sucr = [selected_trials for i_fly, selected_trials in enumerate(all_selected_trials)
    #                             if "sucr" in conditions[i_fly]]
    roi_select_strings = ["210719 fly 2", "210719 fly 1", "210721 fly 3", "210723 fly 1"]
    with PdfPages(os.path.join(OUT_PATH, "210910_dff_maps.pdf")) as pdf:
        for i_fly, (fly_dir, selected_trials, selected_trials_old, condition) \
                in enumerate(zip(fly_dirs, all_selected_trials, all_selected_trials_old, conditions)):
            # if i_fly != 17:
            #     continue
            if len(selected_trials) == len(selected_trials_old):
                continue
            if not any([select_str in condition for select_str in roi_select_strings]):
                USE_ROIS = False
            else:
                USE_ROIS = True
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

            fly_processed_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER)
            roi_center_file = os.path.join(fly_processed_dir, "ROI_centers_rem_dupl.txt")
            # roi_centers = rois.read_roi_center_file(roi_center_file)
            roi_mask_out_dir = os.path.join(fly_processed_dir, "ROI_mask.tif")

            trial_names = []

            rests = []
            walks = []
            twop_dfs = []
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
                green_denoised_dir = os.path.join(processed_dir, "green_denoised_t1_corr.tif")
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
                    if USE_ROIS:
                        twop_df = rois.get_roi_signals_df(stack=green_denoised_dir,
                                        roi_center_filename=roi_center_file,
                                        index_df=twop_df,
                                        mask_out_dir=roi_mask_out_dir) 
                    
                    rest_cleaned = clean_rest(twop_df["rest"].values, N_clean=16*5)
                    twop_df["rest_cleaned"] = rest_cleaned

                    twop_df.to_pickle(twop_out_dir)
                    twop_dfs.append(twop_df)

                rest = np.argwhere(twop_df["rest_cleaned"].to_numpy()).flatten()
                rests.append(rest)
                walk = np.argwhere(twop_df["walk"].to_numpy()).flatten()
                walks.append(walk)
            continue  # TODO
            if USE_ROIS:
                twop_dfs = get_norm_dfs(twop_dfs)
            if MAKE_MAPS:
                dffs = [os.path.join(trial_dir, "processed", "dff_denoised_t1_corr.tif")
                    for trial_dir in trial_dirs]
                dff_map = make_dff_maps(dffs, walks, rests, perc=0.99,
                                        map_dict_out_dir=os.path.join(fly_processed_dir, "dff_maps.pkl"),
                                        trial_names=trial_names)
                fig = plot_dff_maps(dff_map, fly_dir)
                pdf.savefig(fig)
                plt.close(fig)  
            if MAPS_ONLY:
                continue

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
            
            if USE_ROIS:
                text = [trial_name for i_trial, trial_name in enumerate(trial_names)
                        if i_trial in i_trials]
                roi_mask = get_stack(roi_mask_out_dir)
                if os.path.isfile(os.path.join(fly_processed_dir, "roi_dff_video.pkl")) and not OVERWRITE_DF_VIDEO:
                    with open(os.path.join(fly_processed_dir, "roi_dff_video.pkl"), "rb") as f:
                        save_dict = pickle.load(f, protocol=4)
                    assert save_dict["trials"] == text
                    dffs = save_dict["roi_video"]
                else:
                    dffs = [make_roi_dff_video_data(df.filter(regex="norm").values, roi_mask=roi_mask)
                            for i_trial, df in enumerate(twop_dfs) if i_trial in i_trials]
                    
                    save_dict = {"roi_video": dffs, "trials": text}
                    with open(os.path.join(fly_processed_dir, "roi_dff_video.pkl"), "wb") as f:
                        pickle.dump(save_dict, f, protocol=4)
                mask = None
                video_names = ["walking_snippets_ROI", "resting_snippets_ROI"]
                vmax = 100
            else:
                dffs = [os.path.join(trial_dir, "processed", "dff_denoised_t1_corr.tif")
                    for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in i_trials]  # dff_denoised_t1.tif
                try:
                    mask = get_stack(os.path.join(fly_processed_dir, "dff_mask_raw.tif")) > 0  # dff_mask_denoised_t1.tif
                except:
                    mask = None
                video_names = ["walking_snippets", "resting_snippets"]
                vmax = None
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

            if "210901" in condition:
                vmax = 500

            print("===== making videos")
            if all([len(walk)==0 for walk in walks]):
                print("No walking detected. Omitting video.")
            else:
                videos.make_multiple_video_raw_dff_beh(dffs=dffs,
                                                trial_dirs=trial_dirs,
                                                out_dir=fly_processed_dir,
                                                video_name=video_names[0],
                                                beh_dirs=beh_trial_dirs,
                                                sync_dirs=sync_trial_dirs,
                                                camera=6,
                                                stack_axes=[0, 1],
                                                greens=green_dirs,
                                                reds=red_dirs,
                                                vmin=0,
                                                vmax=vmax,
                                                pmin=None,
                                                pmax=99,
                                                share_lim=dff_video_share_lim,
                                                log_lim=dff_video_log_lim,
                                                share_mask=True,
                                                blur=0, mask=mask, crop=None,
                                                text=text,
                                                downsample=None,
                                                select_frames=walks)
                dir_name = os.path.join(TARGET_DIR,condition.replace(" ","_")+"_"+video_names[0]+".mp4")
                print("Copying to ", dir_name)
                copyfile(os.path.join(fly_processed_dir, video_names[0]+".mp4"), dir_name)
                gc.collect()

            videos.make_multiple_video_raw_dff_beh(dffs=dffs,
                                            trial_dirs=trial_dirs,
                                            out_dir=fly_processed_dir,
                                            video_name=video_names[1],
                                            beh_dirs=beh_trial_dirs,
                                            sync_dirs=sync_trial_dirs,
                                            camera=6,
                                            stack_axes=[0, 1],
                                            greens=green_dirs,
                                            reds=red_dirs,
                                            vmin=0,
                                            vmax=vmax,
                                            pmin=None,
                                            pmax=99,
                                            share_lim=dff_video_share_lim,
                                            log_lim=dff_video_log_lim,
                                            share_mask=True,
                                            blur=0, mask=mask, crop=None,
                                            text=text,
                                            downsample=None,
                                            select_frames=rests)
            dir_name = os.path.join(TARGET_DIR, condition.replace(" ", "_")+"_"+video_names[1]+".mp4")
            print("Copying to ", dir_name)
            copyfile(os.path.join(fly_processed_dir, video_names[1]+".mp4"), dir_name)

            gc.collect()


if __name__ == "__main__":
    main()

