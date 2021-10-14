import os, sys
from copy import deepcopy

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load, utils
from twoppp.dff import find_dff_mask
from twoppp.plot.videos import make_video_dff, make_multiple_video_dff
from twoppp.pipeline import PreProcessFly, PreProcessParams

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
    params.make_dff_videos = False  # TODO: make the videos
    params.make_summary_stats = True

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.post_com_crop = True
    params.post_com_crop_values = [64, 80]  # manually assigned for each fly
    params.save_motion_field = False
    params.motion_field = "w.npy"
    params.ofco_verbose = True

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

    params.green_denoised = "green_denoised_t1.tif"
    params.dff = "dff_denoised_t1.tif"
    params.dff_baseline = "dff_baseline_denoised_t1.tif"
    params.dff_mask = "dff_mask_denoised_t1.tif"
    params.dff_video_name = "dff_denoised_t1"
    params.dff_beh_video_name = "dff_beh"
    


    fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
                os.path.join(load.NAS2_DIR_LH, "210723", "fly2")   # high caff
                ]
    all_selected_trials = [[1,4,5,8,10,12],
                           [1,5,6,9,11,12]]
    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials) in \
        enumerate(zip(fly_dirs, all_selected_trials)):
        params = deepcopy(params_copy)   

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        # apply warping to green, denoise, compute dff
        preprocess.run_all_trials()


            
            

       
    
