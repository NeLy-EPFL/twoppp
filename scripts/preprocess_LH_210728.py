import os, sys
from copy import deepcopy

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils
from longterm.dff import find_dff_mask
from longterm.plot.videos import make_video_dff, make_multiple_video_dff
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

    params.green_denoised = "green_denoised_t1.tif"
    params.dff = "dff_denoised_t1.tif"
    params.dff_baseline = "dff_baseline_denoised_t1.tif"
    params.dff_mask = "dff_mask_denoised_t1.tif"
    params.dff_video_name = "dff_denoised_t1"
    params.dff_beh_video_name = "dff_beh"

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

    fly_dirs = [
        os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # high caff
        os.path.join(load.NAS2_DIR_LH, "210721", "fly3"),  # high caff
        os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
        os.path.join(load.NAS2_DIR_LH, "210723", "fly2"),  # high caff
        os.path.join(load.NAS2_DIR_LH, "210718", "fly2"),  # starv

        os.path.join(load.NAS2_DIR_LH, "210719", "fly1"),  # sucr
        os.path.join(load.NAS2_DIR_LH, "210719", "fly2"),  # starv
        os.path.join(load.NAS2_DIR_LH, "210719", "fly4")   # sucr
        ]
    all_selected_trials = [
        [1,4,5,8,10,12],
        [1,4,5,8,10,12],
        [1,4,5,8,10,12],
        [1,5,6,9,11,12],
        [2,4,5,7],

        [2,4,5,7,8],
        [2,4,5,7],
        [2,4,5,7]
        ]

    conditions = [
        "210722 fly 3 high caff",
        "210721 fly 3 high caff",
        "210723 fly 1 low caff",
        "210723 fly 2 high caff",
        "210718 fly 2 starv",

        "210719 fly 1 sucr",
        "210719 fly 2 starv",
        "210719 fly 4 sucr",
        ]

    to_run = [5, 6, 7]

    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials, condition) in \
        enumerate(zip(fly_dirs, all_selected_trials, conditions)):
        params = deepcopy(params_copy)
        if i_fly not in to_run:
            continue

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")

        # warp, denoise, compute denoised dff, compute summary stats
        preprocess.params.make_dff_videos = False
        preprocess.run_all_trials()

        # currently don't make videos
        if False:
            baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
            # thresholds for denoised dff videos:
            threshold = 0
            mask = baseline > threshold

            utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
            preprocess._make_dff_videos(mask=mask)
            preprocess.params.dff_beh_video_name = "dff_beh_2p"
            preprocess.params.dff_video_share_lim = False
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=[0,1,2,-1])

        # further pre-processing steps
        try:
            preprocess.get_dfs()

            # preprocess.extract_rois()

            preprocess.prepare_pca_analysis(condition=condition, load_df=False, load_pixels=True)
        except:
            pass