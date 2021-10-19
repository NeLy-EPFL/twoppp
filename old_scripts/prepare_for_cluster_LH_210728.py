import os, sys
from copy import deepcopy

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load
# from twoppp.dff import find_dff_mask
# from twoppp.plot.videos import make_video_dff, make_multiple_video_dff
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp import fly_dirs, all_selected_trials, all_selected_trials_old, conditions

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


    params_copy = deepcopy(params)

    COPY_BACK_ONLY = True

    for i_fly, (fly_dir, selected_trials, selected_trials_old) in \
        enumerate(zip(fly_dirs, all_selected_trials, all_selected_trials_old)):
        if len(selected_trials) == len(selected_trials_old):
            continue
        if COPY_BACK_ONLY:
            print("COPYING BACK FLY: ", fly_dir)
            stream = os.system(". ../longterm/register/copy_from_cluster.sh " + fly_dir)
            continue
        params = deepcopy(params_copy)

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile",
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        # convert raw to tiff and perform COM registration
        preprocess.run_all_trials()

        stream = os.system(". ../longterm/register/copy_to_cluster.sh " + fly_dir)
