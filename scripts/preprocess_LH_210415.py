import os, sys

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load
from longterm.plot.videos import make_video_dff, make_multiple_video_dff
from longterm.pipeline import PreProcessFly, PreProcessParams

if __name__ == "__main__":

    params = PreProcessParams()
    params.breadth_first = True
    params.overwrite = False

    params.use_warp = True
    params.use_denoise = False
    params.use_dff = True
    params.use_df3d = False
    params.use_df3dPostProcess = False
    params.use_behaviour_classifier = False

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.save_motion_field = False

    # dff params

    params.green_denoised = params.green_com_warped

    params.dff_common_baseline = True
    params.dff_baseline_blur = 5
    params.dff_baseline_med_filt = 5
    params.dff_baseline_blur_pre = False
    params.dff_baseline_mode = "convolve"
    params.dff_baseline_length = 20
    params.dff_baseline_quantile = 0.95
    params.dff_use_crop=[128, 608, 80, 400]
    params.dff_manual_add_to_crop = 20
    params.dff_blur = 0

    """
    fly_dir = os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly2")
    trial_dirs = [os.path.join(fly_dir, "cs_003"),
                os.path.join(fly_dir, "cs_water"),
                os.path.join(fly_dir, "cs_water_after_001")]
    
    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs=trial_dirs)
    preprocess.run_all_trials()
    """
    fly_dir = os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly3")
    trial_dirs = [os.path.join(fly_dir, "cs_002"),
                os.path.join(fly_dir, "cs_caff"),
                os.path.join(fly_dir, "cs_caff_after"),
                os.path.join(fly_dir, "cs_caff_after_003"),
                os.path.join(fly_dir, "cs_caff_after_004"),
                os.path.join(fly_dir, "cs_caff_after_006")]

    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs=trial_dirs)
    # preprocess.run_all_trials()
    preprocess.params.overwrite = True
    preprocess.params.dff_baseline_exclude_trials = [False, True, False, False, True, False]
    # preprocess._compute_dff_alltrials()
    # preprocess.params.overwrite = False
    """
    for i_trial, trial_dir in enumerate(trial_dirs):
        make_video_dff(dff=os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff),
                       out_dir=preprocess.trial_processed_dirs[i_trial],
                       video_name="dff",
                       trial_dir=trial_dir,
                       vmin=0, vmax=250, blur=1)
    """
    make_multiple_video_dff(dffs=[os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff) for i_trial in range(len(trial_dirs))],
                            out_dir=preprocess.fly_processed_dir,
                            video_name="dff_multiple", trial_dir=trial_dirs[0],
                            vmin=0, vmax=250, blur=1)
    

    fly_dir = os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly2")
    trial_dirs = [os.path.join(fly_dir, "cs_003"),
                os.path.join(fly_dir, "cs_water"),
                os.path.join(fly_dir, "cs_water_after_001"),
                os.path.join(fly_dir, "cs_water_after_003")]

    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs=trial_dirs)
    # preprocess.run_all_trials()
    preprocess.params.overwrite = True
    preprocess.params.dff_baseline_exclude_trials = [False, False, False, True]
    # preprocess._compute_dff_alltrials()
    # preprocess.params.overwrite = False
    """
    for i_trial, trial_dir in enumerate(trial_dirs):
        make_video_dff(dff=os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff),
                       out_dir=preprocess.trial_processed_dirs[i_trial],
                       video_name="dff",
                       trial_dir=trial_dir,
                       vmin=0, vmax=250, blur=1)
    """
    make_multiple_video_dff(dffs=[os.path.join(preprocess.trial_processed_dirs[i_trial], preprocess.params.dff) for i_trial in range(len(trial_dirs))],
                            out_dir=preprocess.fly_processed_dir,
                            video_name="dff_multiple", trial_dir=trial_dirs[0],
                            vmin=0, vmax=250, blur=1)
