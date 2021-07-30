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

    params.use_warp = False
    params.use_denoise = False
    params.use_dff = False
    params.use_df3d = False
    params.use_df3dPostProcess = False
    params.use_behaviour_classifier = False
    params.select_trials = False
    params.cleanup_files = False
    params.make_dff_videos = False
    params.make_summary_stats = False

    params.i_ref_trial = 0
    params.i_ref_frame = 0
    params.use_com = True
    params.post_com_crop = True
    params.post_com_crop_values = [64, 80]  # manually assigned for each fly
    params.save_motion_field = False
    params.ofco_verbose = True


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
        # convert raw to tiff and perform COM registration
        preprocess.run_all_trials()

            
            

       
    
