import os, sys

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

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.post_com_crop = True
    params.post_com_crop_values = None  # manually assigned for each fly
    params.save_motion_field = False
    params.ofco_verbose = True

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
    params.dff_min_baseline = None
    params.dff_baseline_exclude_trials = None

    params.dff_video_pmin = None
    params.dff_video_vmin = 0
    params.dff_video_pmax = 99
    params.dff_video_vmax = None

    fly_dirs = [os.path.join(load.NAS_DIR, "LH", "210423_caffeine", "J1M5_fly2"),
                os.path.join(load.NAS_DIR, "LH", "210427_caffeine", "J1M5_fly1")]
    all_trial_dirs = [
        [os.path.join(fly_dirs[0], "cs_001"),
        os.path.join(fly_dirs[0], "cs_caff"),
        os.path.join(fly_dirs[0], "cs_caff_after"),
        os.path.join(fly_dirs[0], "cs_caff_after_006")],

        [os.path.join(fly_dirs[1], "cs_002"),
        os.path.join(fly_dirs[1], "cs_caff"),
        os.path.join(fly_dirs[1], "cs_caff_after"),
        os.path.join(fly_dirs[1], "cs_caff_after_006")]
        ]

    all_selected_trials = [[1,4,5,11],
                           [2,3,4,10]]

    crops = [[48, 144], [48, 144]]
    sizes = [(320, 448), (320, 448)]

    for i_fly, (fly_dir, trial_dirs, selected_trials, crop, size) in \
        enumerate(zip(fly_dirs, all_trial_dirs, all_selected_trials, crops, sizes)):

        print("Starting preprocessing of fly \n" + fly_dir)
        params.post_com_crop_values = crop
        params.denoise_crop_size = size
        # preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs=trial_dirs)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        if i_fly == 0:
            continue
            # preprocess.params.green_denoised = preprocess.params.green_com_warped
            """
            preprocess.params.dff = "dff_denoised.tif"
            preprocess.params.dff_baseline = "dff_baseline_denoised.tif"
            preprocess.params.dff_mask = "dff_mask_denoised.tif"
            preprocess.params.dff_video_name = "dff_denoised"
            params.dff_min_baseline = 0
            params.dff_baseline_exclude_trials = [True, False, False, False]
            preprocess.params.overwrite = False
            preprocess._compute_dff_alltrials()
            preprocess.params.overwrite = False
            preprocess.params.dff_baseline = "dff_baseline.tif"
            preprocess._make_dff_videos(mask=True)
            preprocess.params.dff_baseline = "dff_baseline_denoised.tif"
            preprocess.params.overwrite = False
            """
            preprocess.params.dff = "dff_denoised.tif"
            preprocess.params.dff_baseline = "dff_baseline_denoised.tif"
            preprocess.params.dff_mask = "dff_mask_denoised.tif"
            preprocess.params.dff_video_name = "dff_denoised_difflim"
            preprocess.params.dff_video_share_lim = False
            baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
            mask = baseline > 3  # 3 for denoised, 50 for not denoised
            utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
            # preprocess._make_dff_videos(mask=mask)
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask)
        else:
            # continue
            """
            params.dff_min_baseline = 0

            tmp = preprocess.params.green_denoised
            preprocess.params.green_denoised = preprocess.params.green_com_warped
            preprocess.run_all_trials()

            preprocess.params.green_denoised = tmp
            preprocess.params.dff = "dff_denoised.tif"
            preprocess.params.dff_baseline = "dff_baseline_denoised.tif"
            preprocess.params.dff_mask = "dff_mask_denoised.tif"
            preprocess.params.dff_video_name = "dff_denoised"
            preprocess.run_all_trials()
            """
            # preprocess._compute_dff_alltrials()
            # preprocess.params.dff_baseline = "dff_baseline.tif"
            # preprocess._make_dff_videos(mask=True)
            # preprocess.params.dff_baseline = "dff_baseline_denoised.tif"

            preprocess.params.dff = "dff_denoised.tif"
            preprocess.params.dff_baseline = "dff_baseline_denoised.tif"
            preprocess.params.dff_mask = "dff_mask_denoised.tif"
            preprocess.params.dff_video_name = "dff_denoised"
            baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
            mask = baseline > 10  # 10 for denoised, 50 for not denoised
            utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
            # preprocess._make_dff_videos(mask=mask)
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask)


            
            

       
    
