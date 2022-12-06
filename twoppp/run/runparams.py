"""
sub-module to define parameters for running pre-processing with.

to adapt the parameters for your use:
1. create a dictionary for your user and define the paths and parameters accordingly
2. set CURRENT_USER equal to the dictionary you just created
3. set the run_params according to your use. The major parameters are definde below.
    For more parameters available, e.g. file names,
    see the class definition in the twoppp/pipeline.py file.

also see README.md in the twoppp/run folder for run instructions
"""
import os

from twoppp import load
from twoppp.pipeline import PreProcessParams

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))

USER_JB = {
    "initials": "JB",
    "labserver": os.path.join(load.LABSERVER_DIR, "BRAUN_Jonas", "Experimental_data", "2p"),
    "nas": os.path.join(load.NAS_DIR, "JB"),
    "nas2": os.path.join(load.NAS2_DIR, "JB"),
    # will copy generated videos to this folder
    "video_dir": os.path.join(load.NAS2_DIR, "JB", "_videos"),
    # user name
    "name": "Jonas Braun",
    # under which e-mail to receive status messages
    "email": "nelydebugging@outlook.com",
    "send_emails": True,
    # where the FIDIS scratch directory is mounted locally
    "scratch_dir": "/mnt/scratch/jbraun",
    # whether to check on the scratch directory if files are present or not
    "ignore_scratch": False,
    # whether to ssh into the 2plinux machine to check whether some data might not yet be copied
    "check_2plinux_trials": True,
    # the IP address of the linux computer used for recording
    "2p_linux_ip": "128.178.198.12",
    # the user name of the linuc computer used for recording
    "2p_linux_user": "dalco",
    # which camera should be used for fictrac
    "fictrac_cam": 3,
    # which camera should be used for making summary videos
    "video_cam": 5,
    # which 2pscope you're using
    "2p_scope": 2,
    # fill this file with fly_dirs that should be processed
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    # where to store which tasks are currently running
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
    # whether to check if a task is already running in the _tasks_running.txt before starting it
    "check_tasks_running": False,
}

USER_FH = {
    "initials": "FH",
    "labserver": os.path.join(load.LABSERVER_DIR, "HURTAK_Femke", "Experimental_data", "2p"),
    "nas": os.path.join(load.NAS_DIR, "FH"),
    "nas2": os.path.join(load.NAS2_DIR, "FH"),
    # will copy generated videos to this folder
    "video_dir": os.path.join(load.NAS2_DIR, "FH", "_videos"),
    # user name
    "name": "Femke Hurtak",
    # under which e-mail to receive status messages
    "email": "nelydebugging@outlook.com",
    "send_emails": True,
    # where the FIDIS scratch directory is mounted locally
    "scratch_dir": "/mnt/scratch/hurtak",
    # whether to check on the scratch directory if files are present or not
    "ignore_scratch": False,
    # whether to ssh into the 2plinux machine to check whether some data might not yet be copied
    "check_2plinux_trials": True,
    # the IP address of the linux computer used for recording
    "2p_linux_ip": "128.178.198.12",
    # the user name of the linuc computer used for recording
    "2p_linux_user": "dalco",
    # which camera should be used for fictrac
    "fictrac_cam": 3,
    # which camera should be used for making summary videos
    "video_cam": 5,
    # which 2pscope you're using
    "2p_scope": 2,
    # fill this file with fly_dirs that should be processed
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    # where to store which tasks are currently running
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
    # whether to check if a task is already running in the _tasks_running.txt before starting it
    "check_tasks_running": False,
}

USER_JSP_1ST_SCOPE = {
    "initials": "JSP",
    "labserver": os.path.join(load.LABSERVER_DIR, "PHELPS_Jasper", "data", "2p"),
    "nas": os.path.join(load.NAS_DIR, "JSP"),
    "nas2": os.path.join(load.NAS2_DIR, "JSP"),
    # will copy generated videos to this folder
    "video_dir": os.path.join(load.NAS2_DIR, "JSP", "_videos"),
    # user name
    "name": "Jasper Phelps",
    # under which e-mail to receive status messages
    "email": "nelydebugging@outlook.com",
    "send_emails": False,
    # where the FIDIS scratch directory is mounted locally
    "scratch_dir": "/mnt/scratch/phelps",
    # whether to check on the scratch directory if files are present or not
    "ignore_scratch": False,
    # whether to ssh into the 2plinux machine to check whether some data might not yet be copied
    "check_2plinux_trials": False,
    # the IP address of the linux computer used for recording
    "2p_linux_ip": "128.178.198.189",
    # the user name of the linux computer used for recording
    "2p_linux_user": "dalco",
    # which camera should be used for fictrac
    "fictrac_cam": 4,
    # which camera should be used for making summary videos
    "video_cam": 2,
    # which 2pscope you're using
    "2p_scope": 1,
    # fill this file with fly_dirs that should be processed
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    # where to store which tasks are currently running
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
    # whether to check if a task is already running in the _tasks_running.txt before starting it
    "check_tasks_running": False,
}
USER_JSP_2ND_SCOPE = {
    "initials": "JSP",
    "labserver": os.path.join(load.LABSERVER_DIR, "PHELPS_Jasper", "data", "2p"),
    "nas": os.path.join(load.NAS_DIR, "JSP"),
    "nas2": os.path.join(load.NAS2_DIR, "JSP"),
    # will copy generated videos to this folder
    "video_dir": os.path.join(load.NAS2_DIR, "JSP", "_videos"),
    # user name
    "name": "Jasper Phelps",
    # under which e-mail to receive status messages
    "email": "nelydebugging@outlook.com",
    "send_emails": False,
    # where the FIDIS scratch directory is mounted locally
    "scratch_dir": "/mnt/scratch/phelps",
    # whether to check on the scratch directory if files are present or not
    "ignore_scratch": False,
    # whether to ssh into the 2plinux machine to check whether some data might not yet be copied
    "check_2plinux_trials": False,
    # the IP address of the linux computer used for recording
    "2p_linux_ip": "128.178.198.12",
    # the user name of the linux computer used for recording
    "2p_linux_user": "dalco",
    # which camera should be used for fictrac
    "fictrac_cam": 3,
    # which camera should be used for making summary videos
    "video_cam": 5,
    # which 2pscope you're using
    "2p_scope": 2,
    # fill this file with fly_dirs that should be processed
    "txt_file_to_process": os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt"),
    # where to store which tasks are currently running
    "txt_file_running": os.path.join(LOCAL_DIR, "_tasks_running.txt"),
    # whether to check if a task is already running in the _tasks_running.txt before starting it
    "check_tasks_running": False,
}


# Specify your user either by changing DEFAULT_USER in the line below, or add:
#   export TWOPPP_USER=XYZ
# to your shell rc file (e.g. ~/.bashrc), replacing XYZ with just your initials, e.g.
#   export TWOPPP_USER=JSP
DEFAULT_USER = USER_JB

try:
    CURRENT_USER = eval('USER_{}'.format(os.environ['TWOPPP_USER']))
except KeyError:
    # If TWOPPP_USER is not set, use default
    CURRENT_USER = DEFAULT_USER
except NameError:
    # If TWOPPP_USER is set but that user's settings can't be found
    raise NameError('Environment variable TWOPPP_USER set to {x} but USER_{x} is '
                    'not defined in runparams.py'.format(x=os.environ['TWOPPP_USER']))


global_params = PreProcessParams()
global_params.genotype = ""

global_params.breadth_first = True
global_params.overwrite = False
global_params.select_trials = False

global_params.twoway_align = False
global_params.i_ref_trial = 0
global_params.i_ref_frame = 0
global_params.post_com_crop = True
global_params.post_com_crop_values = [0, 0]  # manually assigned for each fly

global_params.ball_tracking = "fictrac"
global_params.add_df3d_to_df = False

global_params.denoise_crop_size = (320, 736)
global_params.denoise_correct_illumination_leftright = False

global_params.dff_common_baseline = False

global_params.default_video_camera = CURRENT_USER["video_cam"]
global_params.behaviour_as_videos = True
global_params.twop_scope = CURRENT_USER["2p_scope"]

# select all False because they will be manually selected in the different Tasks
global_params.use_com = False
global_params.use_warp = False
global_params.use_denoise = False
global_params.use_dff = False
global_params.make_summary_stats = False
global_params.use_df3d = False
global_params.use_df3dPostProcess = False
global_params.use_behaviour_classifier = False
global_params.cleanup_files = False
global_params.make_dff_videos = False
