import os, sys
from copy import deepcopy

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load
# from longterm.dff import find_dff_mask
# from longterm.plot.videos import make_video_dff, make_multiple_video_dff
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


    fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # high caff
                os.path.join(load.NAS2_DIR_LH, "210721", "fly3"),  # high caff
                os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
                os.path.join(load.NAS2_DIR_LH, "210723", "fly2"),  # high caff

                os.path.join(load.NAS2_DIR_LH, "210802", "fly1"),  # lowcaff
                os.path.join(load.NAS2_DIR_LH, "210804", "fly1"),  # low caff
                os.path.join(load.NAS2_DIR_LH, "210804", "fly2"),  # low caff

                os.path.join(load.NAS2_DIR_LH, "210811", "fly2"),  # high caff
                os.path.join(load.NAS2_DIR_LH, "210811", "fly1"),  # starv
                os.path.join(load.NAS2_DIR_LH, "210812", "fly1"),  # starv
                os.path.join(load.NAS2_DIR_LH, "210813", "fly1"),  # sucr
                os.path.join(load.NAS2_DIR_LH, "210818", "fly3"),  # sucr

                os.path.join(load.NAS2_DIR_LH, "210901", "fly1"),  # starv
                os.path.join(load.NAS2_DIR_LH, "210902", "fly2"),  # sucr
                ]
    all_selected_trials = [
        [1,3,4,5,6,7,8,9,10,11,12,13],
        [1,3,4,5,6,7,8,9,10,11,12],
        [1,3,4,5,6,8,9,10,11,12],
        [1,3,5,6,7,8,9,10,11,12,13,14],

        [1,3,4,5,6,7,8,9,11,12],  # 10 exlcuded because CC out of center
        [1,3,4,5,6,7,8,9,10,11,12],
        [1,3,4,5,6,7,8,9,10,11,12],

        [0,2,5,6,7,8,9,10,11,12,13,16],
        [2,4,5,7],
        [2,4,5,7],
        [2,5,6,8],
        [2,4,5,7],

        [2,4,5,7],
        [2,4,5,7],
        ]

    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials) in \
        enumerate(zip(fly_dirs, all_selected_trials)):
        if i_fly != 13:
            continue
        params = deepcopy(params_copy)

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile",
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        # convert raw to tiff and perform COM registration
        preprocess.run_all_trials()

        stream = os.system(". ../longterm/register/copy_to_cluster.sh " + fly_dir)
