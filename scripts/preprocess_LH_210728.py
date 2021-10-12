import os, sys
import numpy as np
from copy import deepcopy
from time import time, sleep
from tqdm import tqdm
import gc

import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import fly_dirs, all_selected_trials, conditions, all_selected_trials_old
from longterm import load, utils
# from longterm.dff import find_dff_mask
# from longterm.plot.videos import make_video_dff, make_multiple_video_dff
from longterm.pipeline import PreProcessFly, PreProcessParams

if __name__ == "__main__":

    params = PreProcessParams()
    params.genotype = "J1xM5"
    
    params.breadth_first = True
    params.overwrite = False

    params.use_warp = True
    params.use_denoise = True
    params.use_dff = True
    params.use_df3d = False
    params.use_df3dPostProcess = False
    params.use_behaviour_classifier = False
    params.select_trials = False
    params.cleanup_files = False
    params.make_dff_videos = True
    params.make_summary_stats = True

    params.green_denoised = "green_denoised_t1_corr.tif"  # "green_denoised_t1.tif"
    params.dff = "dff_denoised_t1_corr.tif"  # "dff_denoised_t1.tif"
    params.dff_baseline = "dff_baseline_denoised_t1_corr.tif"  # "dff_baseline_denoised_t1.tif"
    params.dff_mask = "dff_mask_denoised_t1_corr.tif"  # "dff_mask_denoised_t1.tif"
    params.dff_video_name = "dff_denoised_t1_corr"  # "dff_denoised_t1"
    params.dff_beh_video_name = "dff_beh_corr"  # "dff_beh"

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.post_com_crop = True
    params.post_com_crop_values = [64, 80]  # manually assigned for each fly
    params.save_motion_field = False
    params.ofco_verbose = True
    params.motion_field = "w.npy"

    params.denoise_crop_size = (352, 576)
    params.denoise_train_each_trial = False
    params.denoise_train_trial = 0
    params.denoise_correct_illumination_leftright = True
    params.denoise_final_dir = "denoising_run_correct"

    # dff params

    params.dff_common_baseline = True
    params.dff_baseline_blur = 10
    params.dff_baseline_med_filt = 1
    params.dff_baseline_blur_pre = True
    params.dff_baseline_mode = "convolve"
    params.dff_baseline_length = 10
    params.dff_baseline_quantile = 0.95
    params.dff_use_crop = None   # [128, 608, 80, 400]
    params.dff_manual_add_to_crop = 20
    params.dff_blur = 0
    params.dff_min_baseline = 0
    params.dff_baseline_exclude_trials = None

    params.dff_video_pmin = None
    params.dff_video_vmin = 0
    params.dff_video_pmax = 99
    params.dff_video_vmax = None
    params.dff_video_share_lim = True
    params.default_video_camera = 6

    to_run = range(18)  # 8, 9, 10, 11,12

    SLEEP = 0  # 7200  # seconds before starting
    STEPSIZE = 30
    start_time = time()
    for t in tqdm(range(0, SLEEP, STEPSIZE)):
        sleep(STEPSIZE)

    params_copy = deepcopy(params)
    LOG_LIM_WORKS = True
    OVERWRITE_DFF_BASELINE = True

    for i_fly, (fly_dir, selected_trials, selected_trials_old, condition) in \
        enumerate(zip(fly_dirs, all_selected_trials, all_selected_trials_old, conditions)):
        params = deepcopy(params_copy)
        if i_fly not in to_run:
            continue
        if len(selected_trials) == len(selected_trials_old):
            continue

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile",
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")

        # warp, denoise, compute denoised dff, compute summary stats
        preprocess.params.make_dff_videos = False
        preprocess.params.make_summary_stats = True  #TODO

        if OVERWRITE_DFF_BASELINE:
            os.remove(os.path.join(fly_dir, load.PROCESSED_FOLDER, preprocess.params.dff_baseline))

        preprocess.run_all_trials()  #TODO

        # making videos
        if False:  # False:
            try:
                # baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
                # thresholds for denoised dff videos:
                # threshold = 0
                # mask = baseline > threshold
                mask = utils.get_stack(os.path.join(preprocess.fly_processed_dir, "dff_mask_raw.tif")).astype(bool)  # preprocess.params.dff_mask

                # utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
                # preprocess._make_dff_videos(mask=mask) #TODO
                preprocess.params.dff_beh_video_name = "dff_beh_2p_corr"
                if "caff" in condition:
                    preprocess.params.dff_video_share_lim = False
                    preprocess.params.dff_video_log_lim = [False, False, False, True]
                    last_trial = np.where(["007" in trial for trial in preprocess.trial_dirs])[0]
                    if not last_trial.size:
                        last_trial = np.where(["006" in trial for trial in preprocess.trial_dirs])[0]
                    if not last_trial.size:
                        last_trial = [-1]
                    last_trial = last_trial[0]
                    i_trials = [0,2,3,last_trial]
                    if "210723 fly 2" in condition:
                        i_trials = [1,2,3,last_trial]
                else:
                    preprocess.params.dff_video_share_lim = True
                    preprocess.params.dff_video_log_lim = False
                    i_trials = [0,1,2,-1]
                # preprocess.params.overwrite = True
                if LOG_LIM_WORKS:
                    try:
                        preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=i_trials)
                    except:
                        # LOG_LIM_WORKS = False
                        preprocess.params.dff_video_log_lim = False
                        preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=i_trials)
                else:
                    preprocess.params.dff_video_log_lim = False
                    preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=i_trials)

                # preprocess.params.overwrite = False
            except:
                print("Error while making video for fly", fly_dir)

        
        # further pre-processing steps
        preprocess.get_dfs()
        if False:
            try:
                preprocess.get_dfs()

                # preprocess.extract_rois()
                if "caff" in condition:
                    i_trials = [trial for i_t, trial in enumerate(selected_trials) if i_t != 2]
                    zscore_trials = [0,1]
                else:
                    i_trials = [trial for i_t, trial in enumerate(selected_trials) if i_t != 1]
                    zscore_trials = [0]
                preprocess.prepare_pca_analysis(condition=condition, load_df=False, load_pixels=True,
                                                i_trials=i_trials,
                                                sigma=0, zscore_trials=zscore_trials)
                
            except KeyboardInterrupt as e:
                raise e
            except:
                print(f"Could not prepare for PCA analysis for fly {fly_dir}")

        # raw dff and raw dff videos
        if False:
            preprocess.params.dff_baseline = "dff_baseline_raw.tif"
            preprocess.params.green_denoised = "green_com_warped.tif"
            preprocess.params.dff = "dff_raw.tif"
            preprocess.params.dff_mask = "dff_mask_raw.tif"
            preprocess._compute_dff_alltrials()
            mask = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask)).astype(bool)

            preprocess.params.dff_beh_video_name = "dff_beh_2p_raw"
            if "caff" in condition:
                preprocess.params.dff_video_share_lim = False
                preprocess.params.dff_video_log_lim = [False, False, False, True]
                last_trial = np.where(["007" in trial for trial in preprocess.trial_dirs])[0]
                if not last_trial.size:
                    last_trial = np.where(["006" in trial for trial in preprocess.trial_dirs])[0]
                if not last_trial.size:
                    last_trial = [-1]
                last_trial = last_trial[0]
                i_trials = [0,2,3,last_trial]
                if "210723 fly 2" in condition:
                    i_trials = [1,2,3,last_trial]
            else:
                preprocess.params.dff_video_share_lim = True
                preprocess.params.dff_video_log_lim = False
                i_trials = [0,1,2,-1]
            preprocess.params.overwrite = True
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=i_trials)
            preprocess.params.overwrite = False
            gc.collect()


        # make behavioural video snippets
        if False:
            pass
