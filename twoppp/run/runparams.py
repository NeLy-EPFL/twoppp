import os
from copy import deepcopy

from twoppp import load
from twoppp.pipeline import PreProcessParams

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))

USER_JB = {
    "initials": "JB",
    "labserver": os.path.join(load.LABSERVER_DIR, "BRAUN_Jonas", "Experimental_data", "2p"),
    "nas": os.path.join(load.NAS_DIR, "JB"),
    "nas2": os.path.join(load.NAS2_DIR, "JB"),
    "video_dir": os.path.join(load.NAS2_DIR, "JB", "_videos"),
    "name": "Jonas Braun",
    "email": "nelydebugging@outlook.com",
    "send_emails": True,
    "scratch_dir": "/mnt/scratch",
    "ignore_scratch": False,
    "check_2plinux_trials": True,
    "fictrac_cam": 3,
    "video_cam": 5,
    "2p_scope": 2,
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
}

CURRENT_USER = USER_JB


global_params = PreProcessParams()

global_params.genotype = ""

global_params.breadth_first = True
global_params.overwrite = False

global_params.use_df3d = False
global_params.use_df3dPostProcess = False
global_params.use_behaviour_classifier = False
global_params.select_trials = False
global_params.cleanup_files = False
global_params.make_dff_videos = False

global_params.i_ref_trial = 0
global_params.i_ref_frame = 0
global_params.post_com_crop = True
global_params.post_com_crop_values = [0, 0]  # manually assigned for each fly

global_params.ball_tracking = "fictrac"
global_params.add_df3d_to_df = False

global_params.denoise_crop_size = (320, 736)
global_params.denoise_correct_illumination_leftright = False

global_params.default_video_camera = CURRENT_USER["video_cam"]
global_params.behaviour_as_videos = True
global_params.twop_scope = CURRENT_USER["2p_scope"]

global_params.use_com = False
global_params.use_warp = False
global_params.use_denoise = False
global_params.use_dff = False
global_params.make_summary_stats = False
global_params.use_df3d = False
global_params.use_df3dPostProcess = False