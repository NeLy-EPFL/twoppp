import os,sys
import numpy as np
import pickle

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load, utils
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp.rois import local_correlations

fly_dirs = [os.path.join(load.NAS_DIR_LH, "210415", "J1M5_fly2"),
                os.path.join(load.NAS_DIR_LH, "210423_caffeine", "J1M5_fly2"),
                os.path.join(load.NAS_DIR_LH, "210427_caffeine", "J1M5_fly1"),
                os.path.join(load.NAS2_DIR_LH, "210512", "fly3"),
                os.path.join(load.NAS2_DIR_LH, "210514", "fly1")]

all_selected_trials = [[2,3,5,7],
                       [1,4,5,11],
                       [2,3,4,10],
                       [2,4,5,11],
                       [0,5,6,12]]

for i_fly, (fly_dir, selected_trials) in enumerate(zip(fly_dirs, all_selected_trials)):
    print("fly {}".format(i_fly))
    params = PreProcessParams()
    params.use_warp = False
    params.use_denoise = False
    params.use_dff = False
    params.make_dff_videos = False
    params.dff = "dff_denoised_t1.tif"  # "dff.tif"  # "dff_denoised_t1.tif"
    params.green_denoised = "green_denoised_t1.tif"  # "green_com_warped.tif"  # "green_denoised_t1.tif"

    preprocess = PreProcessFly(fly_dir, params=params, trial_dirs="fromfile", 
                               selected_trials=selected_trials,
                               beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
    # get dff stacks
    dffs = [os.path.join(processed_dir, preprocess.params.dff) for processed_dir in preprocess.trial_processed_dirs]
    dffs = [utils.get_stack(dff) for dff in dffs]

    # compute quantities
    means = [np.mean(dff, axis=0) for dff in dffs]
    mean_diffs = [mean - means[0] for mean in means]
    stds = [np.std(dff, axis=0) for dff in dffs]
    std_diffs = [std - stds[0] for std in stds]
    quants = [np.percentile(dff, 95, axis=0) for dff in dffs]
    quant_diffs = [quant - quants[0] for quant in quants]
    local_corrs = [local_correlations(dff) for dff in dffs]
    local_corr_diffs = [local_corr - local_corrs[0] for local_corr in local_corrs]

    del dffs

    # get green stacks
    greens = [os.path.join(processed_dir, preprocess.params.green_denoised) for processed_dir in preprocess.trial_processed_dirs]
    greens = [utils.get_stack(green) for green in greens]

    # compute quantities
    means_green = [np.mean(green, axis=0) for green in greens]
    mean_diffs_green = [mean - means_green[0] for mean in means_green]
    stds_green = [np.std(green, axis=0) for green in greens]
    std_diffs_green = [std - stds_green[0] for std in stds_green]
    quants_green = [np.percentile(green, 95, axis=0) for green in greens]
    quant_diffs_green = [quant - quants_green[0] for quant in quants_green]
    local_corrs_green = [local_correlations(green) for green in greens]
    local_corr_diffs_green = [local_corr - local_corrs_green[0] for local_corr in local_corrs_green]

    del greens

    # make dictionary
    output_dict = {
        "dff_means": means,
        "dff_mean_diffs": mean_diffs,
        "green_means": means_green,
        "green_mean_diffs": mean_diffs_green,
        "dff_stds": stds,
        "dff_std_diffs": std_diffs,
        "green_stds": stds_green,
        "green_std_diffs": std_diffs_green,
        "dff_quants": quants,
        "dff_quant_diffs": quant_diffs,
        "green_quants": quants_green,
        "green_quant_diffs": quant_diffs_green,
        "dff_local_corrs": local_corrs,
        "dff_local_corr_diffs": local_corr_diffs,
        "green_local_corrs": local_corrs_green,
        "green_local_corr_diffs": local_corr_diffs_green
    }

    output = os.path.join(preprocess.fly_processed_dir, "compare_trials.pkl")  # "compare_trials.pkl" "compare_trials_raw.pkl"
    with open(output, "wb") as f:
        pickle.dump(output_dict, f)



