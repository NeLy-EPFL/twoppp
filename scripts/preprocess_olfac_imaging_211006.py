import os, sys

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from os.path import join
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from longterm import load, utils
from longterm.pipeline import PreProcessFly, PreProcessParams
from longterm.behaviour.fictrac import config_and_run_fictrac
from longterm.behaviour.olfaction import get_sync_signals_olfaction, average_neural_across_repetitions
from longterm.plot.videos import make_all_odour_condition_videos

params = PreProcessParams()
params.genotype = "J1xM5"

params.breadth_first = True
params.overwrite = False

params.use_df3d = False
params.use_df3dPostProcess = False
params.use_behaviour_classifier = False
params.select_trials = False
params.cleanup_files = False
params.make_dff_videos = False

params.i_ref_trial = 0
params.i_ref_frame = 0
params.use_com = True
params.post_com_crop = True
params.post_com_crop_values = [80, 0]  # manually assigned for each fly (Laura: 64, 80)

params.ball_tracking = "fictrac"

params.denoise_crop_size = (320, 736)
params.denoise_correct_illumination_leftright = False

params.default_video_camera = 5
params.behaviour_as_videos = True
params.twop_scope = 2

params.use_warp = False
params.use_denoise = False
params.use_dff = False
params.make_summary_stats = False
params.use_df3d = False
params.use_df3dPostProcess = False

params_copy = deepcopy(params)

if __name__ == "__main__":

    date_dir = join(load.NAS2_DIR_JB, "211005_J1M5")

    fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)
    fly_dirs = fly_dirs[::-1]  # TODO
    all_trial_dirs = load.get_trials_from_fly(fly_dirs, endswith="xz", startswith="0")

    STAGES = ["PREP_CLUSTER", "FICTRAC", "FICTRAC_SIGNALS", "STIM_VIDEOS", "COPY_FROM_CLUSTER",
              "DENOISE&DFF", "DFF_VIDEOS", "CONDITION_AVERAGE", "DF3D"]
    I_STAGE = 8  # TODO: select stage for processing
    STAGE = STAGES[I_STAGE]

    for i_fly, (fly_dir, trial_dirs) in enumerate(zip(fly_dirs, all_trial_dirs)):
        # if i_fly == 1:  # TODO
        #     continue
        if STAGE == "PREP_CLUSTER":
            print("STARTING PREPROCESSING OF FLY: \n" + fly_dir)
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)
            # convert raw to tiff and perform COM registration
            preprocess.run_all_trials()

            print("COPYING TO CLUSTER: ", fly_dir)
            utils.run_shell_command(". " + os.path.join(MODULE_PATH, "longterm", "register",
                                                        "copy_to_cluster.sh") + " " + fly_dir)

        elif STAGE == "FICTRAC":
            config_and_run_fictrac(fly_dir, trial_dirs)

        elif STAGE == "FICTRAC_SIGNALS":
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)
            preprocess.get_dfs()
            for i_trial, trial_dir in enumerate(tqdm(trial_dirs)):
                df = join(trial_dir, load.PROCESSED_FOLDER, preprocess.params.df3d_df_out_dir)
                _ = get_sync_signals_olfaction(
                        trial_dir,
                        sync_out_file=load.PROCESSED_FOLDER+"/sync.pkl",
                        paradigm_out_file=load.PROCESSED_FOLDER+"/paradigm.pkl",
                        index_df=df,
                        df_out_dir=df)

        elif STAGE == "STIM_VIDEOS":
            for i_trial, trial_dir in enumerate(tqdm(trial_dirs)):
                _ = get_sync_signals_olfaction(
                    trial_dir,
                    sync_out_file=load.PROCESSED_FOLDER+"/sync.pkl",
                    paradigm_out_file=load.PROCESSED_FOLDER+"/paradigm.pkl")

            print("MAKING STIM RESPONSE VIDEO: ", fly_dir)
            video_dirs = [join(trial_dir, "behData", "images", "camera_5.mp4")
                          for trial_dir in trial_dirs]
            paradigm_dirs = [join(trial_dir, load.PROCESSED_FOLDER, "paradigm.pkl")
                             for trial_dir in trial_dirs]
            out_dir = join(fly_dir, load.PROCESSED_FOLDER)
            make_all_odour_condition_videos(video_dirs, paradigm_dirs, out_dir,
                                            video_name="stim_responses", frame_range=[-500,1500],
                                            stim_length=1000, frame_rate=None,
                                            size=(120,-1), conditions=["WaterB"])

        elif STAGE == "COPY_FROM_CLUSTER":
            print("COPYING BACK FLY: ", fly_dir)
            utils.run_shell_command(". " + join(MODULE_PATH, "longterm", "register",
                                                "copy_from_cluster.sh") + " " + fly_dir)

        elif STAGE == "DENOISE&DFF":
            params.use_warp = True
            params.use_denoise = True
            params.use_dff = True
            params.make_summary_stats = True

            params_copy = deepcopy(params)

            print("STARTING PREPROCESSING OF FLY: \n" + fly_dir)
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)
            # finalise motion correction, denoise, compute dff
            preprocess.run_all_trials()

        elif STAGE == "DFF_VIDEOS":
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)
            preprocess.params.dff_beh_video_name = "dff_beh"
            preprocess.params.dff_video_share_lim = True
            preprocess.params.dff_video_log_lim = False
            i_trials = [0,2,4]
            print("MAKING VIDEO OF FLY: \n" + fly_dir)
            preprocess._make_dff_behaviour_video_multiple_trials(include_2p=True, i_trials=i_trials,
                                                                 select_frames=[np.arange(30*16,150*16)]*3)

        elif STAGE == "CONDITION_AVERAGE":
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)

            beh_dfs = [os.path.join(trial_processed_dir, preprocess.params.df3d_df_out_dir)
                                    for trial_processed_dir in preprocess.trial_processed_dirs]
            to_average = [os.path.join(trial_processed_dir, preprocess.params.green_denoised)
                                    for trial_processed_dir in preprocess.trial_processed_dirs]
            output_dir=os.path.join(preprocess.fly_processed_dir, "mean_denoised_green.tif")
            _ = average_neural_across_repetitions(beh_dfs, to_average, output_dir=output_dir)
            """
            to_average = [os.path.join(trial_processed_dir, preprocess.params.green_denoised) 
                                    for trial_processed_dir in preprocess.trial_processed_dirs]
            output_dir=os.path.join(preprocess.fly_processed_dir, "mean_dff.tif")
            _ = average_neural_across_repetitions(beh_dfs, to_average, output_dir=output_dir)
            """
        elif STAGE == "DF3D":
            params.use_df3d = True
            params.use_df3dPostProcess = True

            params_copy = deepcopy(params)

            print("STARTTING PREPROCESSING OF FLY: \n" + fly_dir)
            preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy, trial_dirs=trial_dirs)
            # run df3d and df3d postprocessing
            preprocess._pose_estimate(trial_dirs=preprocess.beh_trial_dirs[0:1])
            # preprocess.run_all_trials()  # TODO: change this back to default
