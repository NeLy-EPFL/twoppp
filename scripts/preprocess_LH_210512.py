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
    params.dff_min_baseline = 0
    params.dff_baseline_exclude_trials = None

    params.dff_video_pmin = None
    params.dff_video_vmin = 0
    params.dff_video_pmax = 99
    params.dff_video_vmax = None
    params.dff_video_share_lim = True
    params.default_video_camera = 6

    fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210512", "fly3"),  # water
                os.path.join(load.NAS2_DIR_LH, "210514", "fly1"),  # caff -> bad
                os.path.join(load.NAS2_DIR_LH, "210519", "fly1"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210521", "fly1"),  # nofeed
                os.path.join(load.NAS2_DIR_LH, "210524", "fly1"),  # bad

                os.path.join(load.NAS2_DIR_LH, "210526", "fly2"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210527", "fly4"),  # nofeed
                os.path.join(load.NAS2_DIR_LH, "210531", "fly2"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210602", "fly2"),  # sucr
                os.path.join(load.NAS2_DIR_LH, "210603", "fly2"),  # sucr

                os.path.join(load.NAS2_DIR_LH, "210604", "fly3"),  # nofeed
                os.path.join(load.NAS2_DIR_LH, "210616", "fly2"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210618", "fly5"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210715", "fly3"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210718", "fly2"),  # starv

                os.path.join(load.NAS2_DIR_LH, "210721", "fly3"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # caff
                os.path.join(load.NAS2_DIR_LH, "210719", "fly1")  # sucr
                ]
    all_selected_trials = [[0,2,3,4,5,8,9,11,13],  # [2,4,5,11]  # [0,2,3,4,5,8,11]
                           [0,5,6,8,12,14],  # 0,5,6,12
                           [3,5,8,12],
                           [3,4,7,12],
                           [2,4,7,11],

                           [2,4,7,11],
                           [3,4,6,10],
                           [4,6,9,12],
                           [3,4,6,10],
                           [1,4,6,10],

                           [3,5,7,10],
                           [2,4,7,11,13,16],
                           [2,5,6,7,8,9,10],
                           [2,4,5,8,10,11],
                           [2,4,5,7],
                           
                           [1,4,5,8,10,12],
                           [1,4,5,8,10,12],
                           [2,4,5,7,8]]
    ref_frames = [2268, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 
                  0, 0, 0, 2000, 0,
                  0, 0, 0]
    crops = [[32, 48], [0, 0], [64, 80], [64, 80], [64, 80], 
             [64, 80], [64, 80], [64, 80], [64, 80], [64, 80], 
             [64, 80], [64, 80], [64, 80], [64, 80], [64, 80],
             [64, 80], [64, 80], [64, 80]]
    sizes = [(352, 640), (352, 640), (320, 576), (352, 576), (352, 576), 
             (352, 576), (352, 576), (352, 576), (352, 576), (352, 576), 
             (352, 576), (352, 576), (352, 576), (352, 576), (352, 576),
             (352, 576), (352, 576), (352, 576)]
    exclude_baselines = [[False, False, False, False, False, False, False],
                         [False, False, False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],

                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],

                         [False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False],
                         [False, False, False, False],
                         
                         [False, False, False, False, False, False],
                         [False, False, False, False, False, False],
                         [False, False, False, False, False]]

    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials, crop, size, exclude_baseline, ref_frame) in \
        enumerate(zip(fly_dirs, all_selected_trials, crops, sizes, exclude_baselines, ref_frames)):
        params = deepcopy(params_copy)
        if i_fly < 17:
            continue
            # pass
        else:
            # continue
            pass
        if i_fly == 1:
            params.post_com_crop = False
        

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
        # warp and compute not denoised dff
        """
        preprocess.params.green_denoised = preprocess.params.green_com_warped
        preprocess.params.make_dff_videos = False
        preprocess.run_all_trials()
        
        preprocess.params.dff_beh_video_name = "dff_beh_raw"
        baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
        if i_fly == 0:
            mask = baseline > 128
        elif i_fly == 1:
            mask = baseline > 0
        elif i_fly == 2:
            mask = baseline > 0
        utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
        # preprocess._make_dff_videos(mask=mask)
        # preprocess._make_dff_behaviour_video_multiple_trials(mask=mask)
        """
        # warp and compute denoised dff 
        preprocess.params.green_denoised = "green_denoised_t1.tif"
        preprocess.params.dff = "dff_denoised_t1.tif"
        preprocess.params.dff_baseline = "dff_baseline_denoised_t1.tif"
        preprocess.params.dff_mask = "dff_mask_denoised_t1.tif"
        preprocess.params.dff_video_name = "dff_denoised_t1"
        preprocess.params.dff_beh_video_name = "dff_beh"
        if i_fly == 0:
            print("compute summary stats for fly 0")
            preprocess.params.overwrite = True
            preprocess._compute_summary_stats()
            break
        if i_fly >= 2:
            preprocess.params.denoise_train_each_trial = False
            preprocess.params.denoise_train_trial = 0

        preprocess.params.make_dff_videos = False
        preprocess.run_all_trials()  # denoise and compute dff
        """
        baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_baseline))
        # thresholds for denoised dff videos:
        if i_fly == 0:  
            threshold = 10  # 5-32
        elif i_fly == 1:  
            threshold = 10  # 10-32
        elif i_fly == 2:
            threshold = 7.5  # 7.5-12.5
        elif i_fly == 3:
            threshold = 5  # 2.5-7.5
        elif i_fly == 4:
            threshold = 0
        elif i_fly >= 5:
            threshold = 0
        mask = baseline > threshold

        utils.save_stack(os.path.join(preprocess.fly_processed_dir, preprocess.params.dff_mask), mask)
        preprocess._make_dff_videos(mask=mask)
        preprocess.params.dff_beh_video_name = "dff_beh_2p"
        if i_fly >= 12:
            preprocess.params.dff_video_share_lim = False
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True, i_trials=[0,1,2,5])
            
        else:
            preprocess._make_dff_behaviour_video_multiple_trials(mask=mask, include_2p=True)
        preprocess.params.dff_beh_video_name = "dff_beh"
        # preprocess._make_dff_behaviour_video_multiple_trials(mask=mask)
        """
        
        

            
            

       
    
