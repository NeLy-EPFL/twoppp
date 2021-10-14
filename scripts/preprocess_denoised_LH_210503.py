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

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.save_motion_field = False

    # dff params

    # params.green_denoised = params.green_com_warped
    params.dff = "dff_denoised.tif"
    params.dff_baseline = "dff_baseline_denoised.tif"

    params.dff_common_baseline = True
    params.dff_min_baseline = 10
    params.dff_baseline_blur = 10
    params.dff_baseline_med_filt = 1
    params.dff_baseline_blur_pre = True
    params.dff_baseline_mode = "convolve"
    params.dff_baseline_length = 10
    params.dff_baseline_quantile = 0.95
    params.dff_use_crop = False  # [128, 608, 80, 400]
    params.dff_manual_add_to_crop = 20
    params.dff_blur = 0
    
    fly_dir = os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly3")
    trial_dirs = [os.path.join(fly_dir, "cs_002"),
                os.path.join(fly_dir, "cs_caff"),
                os.path.join(fly_dir, "cs_caff_after"),
                os.path.join(fly_dir, "cs_caff_after_003"),
                os.path.join(fly_dir, "cs_caff_after_004"),
                os.path.join(fly_dir, "cs_caff_after_006")]

    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                               beh_trial_dirs="fromfile", sync_trial_dirs="fromfile", selected_trials=[2, 3, 4, 7, 8, 10])
    preprocess.params.dff_baseline_exclude_trials = [False, True, False, False, True, False]
    # preprocess.run_all_trials()
    preprocess.params.overwrite = False
    preprocess._compute_dff_alltrials()
    preprocess.params.overwrite = False
    # preprocess._make_dff_behaviour_video_trial(i_trial=0)
    preprocess._make_dff_behaviour_video_multiple_trials(i_trials=[0,1,2,3], mask=None)
    """
    # use the none-denoised baseline for cropping
    baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, "dff_baseline.tif"))  # preprocess.params.dff_baseline))
    # baseline = baseline[params.dff_use_crop[2]:params.dff_use_crop[3], params.dff_use_crop[0]:params.dff_use_crop[1]]
    baseline = baseline[80:400, 48:688]
    mask = find_dff_mask(baseline)
    mask = None
    
    for i_trial, trial_dir in enumerate(trial_dirs):
        make_video_dff(dff=os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff),
                       out_dir=preprocess.trial_processed_dirs[i_trial],
                       video_name="dff_denoised",
                       trial_dir=trial_dir,
                       vmin=0, pmax=99, blur=0,
                       mask=mask, crop=None, text=preprocess.trial_names[i_trial])
    
    make_multiple_video_dff(dffs=[os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff) for i_trial in range(len(trial_dirs))],
                            out_dir=preprocess.fly_processed_dir,
                            video_name="dff_multiple_denoised_same_vmax", trial_dir=trial_dirs[0],
                            vmin=0, pmax=99, blur=0, share_lim=True,
                            mask=mask, share_mask=True, crop=None, text=preprocess.trial_names)
    """


    fly_dir = os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly2")
    trial_dirs = [os.path.join(fly_dir, "cs_003"),
                os.path.join(fly_dir, "cs_water"),
                os.path.join(fly_dir, "cs_water_after_001"),
                os.path.join(fly_dir, "cs_water_after_003")]

    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                               beh_trial_dirs="fromfile", sync_trial_dirs="fromfile", selected_trials=[2, 3, 5, 7])
    preprocess.params.dff_baseline_exclude_trials = [False, False, False, True]
    # preprocess.run_all_trials()
    preprocess.params.overwrite = False
    preprocess._compute_dff_alltrials()
    preprocess.params.overwrite = False
    preprocess._make_dff_behaviour_video_multiple_trials(i_trials=[0,1,2,3], mask=None)
    """
    baseline = utils.get_stack(os.path.join(preprocess.fly_processed_dir, "dff_baseline.tif"))  # preprocess.params.dff_baseline))
    # baseline = baseline[params.dff_use_crop[2]:params.dff_use_crop[3], params.dff_use_crop[0]:params.dff_use_crop[1]]
    baseline = baseline[80:400, 48:688]
    mask = find_dff_mask(baseline)
    mask = None
    
    for i_trial, trial_dir in enumerate(trial_dirs):
        make_video_dff(dff=os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff),
                       out_dir=preprocess.trial_processed_dirs[i_trial],
                       video_name="dff_denoised",
                       trial_dir=trial_dir,
                       vmin=0, pmax=99, blur=0,
                       mask=mask, crop=None, text=preprocess.trial_names[i_trial])
    
    make_multiple_video_dff(dffs=[os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff) for i_trial in range(len(trial_dirs))],
                            out_dir=preprocess.fly_processed_dir,
                            video_name="dff_multiple_denoised_same_vmax", trial_dir=trial_dirs[0],
                            vmin=0, pmax=99, blur=0, share_lim=True,
                            mask=mask, share_mask=True, crop=None, text=preprocess.trial_names)
    """

