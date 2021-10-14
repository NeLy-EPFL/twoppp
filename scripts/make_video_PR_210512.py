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
    params.dff_use_crop = None
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

    fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210512", "fly3")
                ]
    all_selected_trials = [[2,4,5,11]  # [0,2,3,4,5,8,11]  # 
                           ]
    ref_frames = [2268]
    crops = [[32, 48]]
    sizes = [(352, 640)]
    exclude_baselines = [[False, False, False, False]]

    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials, crop, size, exclude_baseline, ref_frame) in \
        enumerate(zip(fly_dirs, all_selected_trials, crops, sizes, exclude_baselines, ref_frames)):
        params = deepcopy(params_copy)

        print("Starting preprocessing of fly \n" + fly_dir)
        params.i_ref_frame = ref_frame
        if i_fly == 0 and selected_trials[0] != 2:
            ref_trial = selected_trials.index(2)
            params.i_ref_trial = ref_trial
        else:
            params.i_ref_trial = 0
        params.post_com_crop_values = crop
        params.denoise_crop_size = size
        params.dff_baseline_exclude_trials = exclude_baseline
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        # warp and compute denoised dff 
        preprocess.params.green_denoised = "green_denoised_t1.tif"
        preprocess.params.dff = "dff_denoised_t1.tif"
        preprocess.params.dff_baseline = "dff_baseline_denoised_t1.tif"
        preprocess.params.dff_mask = "dff_mask_denoised_t1.tif"
        preprocess.params.dff_video_name = "dff_denoised_t1"
        preprocess.params.dff_beh_video_name = "dff_beh"

        baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
        # thresholds for denoised dff videos:
        if i_fly == 0:  
            threshold = 10  # 5-32
        mask = baseline > threshold

        utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
        
        preprocess.params.dff_beh_video_name = "dff_beh"
        preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=False)

        

        

            
            

       
    
