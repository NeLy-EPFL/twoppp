"""
sub-module for pose estimation.
Includes functions to prepare a run-script for DeepFly3D, 
run said script and perform post-processing with df3dPostProcessing.
"""
import sys
import os
from shutil import copy
import glob
import pickle

from df3dPostProcessing.df3dPostProcessing import df3dPostProcess

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
TWOPPP_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
sys.path.append(MODULE_PATH)

from twoppp.utils import makedirs_safe, find_file
from twoppp.load import NAS2_DIR_JB, get_trials_from_fly



def prepare_for_df3d(trial_dirs, videos=True, scope=2, tmp_process_dir=None, overwrite=False):
    """Prepare a run script for DeepFly3D depending on whether the data source is videos/images.
    Use the correct camera ordering depending on which set-up the data was acquired.
    Write a .txt file including all the trial dirs in the same folder.

    Parameters
    ----------
    trial_dirs : list
        list of absolute trial directories that are to be processed
    videos : bool, optional
        whether the data source is videos or images, by default True
    scope : int, optional
        which two-photon set-up was used to acquire the data. 1=LH&CLC, 2=FA&JB, by default 2
    tmp_process_dir : string or None, optional
        from which folder to run deepfly3d. If none, will be ../../../tmp, by default None
    overwrite : bool, optional
        whether to overwrite excisting 'pose_result*' files, by default False

    Returns
    -------
    string
        tmp_process_dir

    Raises
    ------
    NotImplementedError
        if scope not in [1,2]
    """
    if not isinstance(trial_dirs, list):
        trial_dirs = [trial_dirs]
    
    tmp_process_dir = os.path.join(MODULE_PATH, "tmp") if tmp_process_dir is None else tmp_process_dir
    makedirs_safe(tmp_process_dir)

    # copy the relevant run script to tmp folder
    run_script = os.path.join(tmp_process_dir, "run_df3d.sh")
    if videos and scope == 2:
        copy(src=os.path.join(BEHAVIOUR_PATH, "run_df3d_videos_2ndscope.sh"),
                 dst=run_script)
    elif scope == 2:
        copy(src=os.path.join(BEHAVIOUR_PATH, "run_df3d_images_2ndscope.sh"),
                 dst=run_script)
    elif videos and scope == 1:
        copy(src=os.path.join(BEHAVIOUR_PATH, "run_df3d_videos_1stscope.sh"),
                 dst=run_script)
    elif scope == 1:
        copy(src=os.path.join(BEHAVIOUR_PATH, "run_df3d_images_1stscope.sh"),
                 dst=run_script)
    else:
        raise NotImplementedError

    folders_dir = os.path.join(tmp_process_dir, "folders.txt")
    with open(folders_dir, "w") as f:
        f.truncate(0)  # make sure all previous entries are deleted
        for trial_dir in trial_dirs:
            images_dir = find_file(trial_dir, "images", "images folder")
            if overwrite or not len(glob.glob(os.path.join(images_dir, "df3d", "pose_result*"))):
                f.write(trial_dir + "\n")

    return tmp_process_dir

def run_df3d(tmp_process_dir):
    """run deepfly3d shell commands using os.system()

    Parameters
    ----------
    tmp_process_dir : string
        directory in which a "run_df3d.sh" script is located to run deepfly3d 
        and a folders.txt file that includes all the trial directories to run.

    Raises
    ------
    FileNotFoundError
        in case the run_df3d.sh script or the folders.txt file cannot be located
    """
    run_script = os.path.join(tmp_process_dir, "run_df3d.sh")
    folders_dir = os.path.join(tmp_process_dir, "folders.txt")
    if not os.path.isfile(run_script) or not os.path.isfile(folders_dir):
        raise FileNotFoundError
    if os.stat(folders_dir).st_size:  # confirm that the folders.txt file is not empty
        os.chdir(tmp_process_dir)
        os.system("pwd")
        os.system("sh run_df3d.sh")

def postprocess_df3d_trial(trial_dir, overwrite=False):
    """run post-processing of deepfly3d data as defined in the df3dPostProcessing package:
    Align with reference fly template and calculate leg angles.

    Parameters
    ----------
    trial_dir : string
        directory of the trial. should contain an "images" folder at some level of hierarchy
    overwrite : bool, optional
        whether to overwrite existing results, by default False
    """
    images_dir = find_file(trial_dir, "images", "images folder")
    pose_result = glob.glob(os.path.join(images_dir, "df3d", "pose_result*"))[0]
    if overwrite or not len(glob.glob(os.path.join(images_dir, "df3d", "joint_angles*"))):
        try:
            mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result, calculate_3d=True, correct_outliers=True)
        except:
            print("New version of df3d post processing did not work. will not correct outliers")
            mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result)
        try:
            aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True, all_body=True)
        except:
            print("New version of df3d post processing did not work. will not align antennal markers")
            aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True)
        path = pose_result.replace('pose_result','aligned_pose')
        with open(path, 'wb') as f:
            pickle.dump(aligned_model, f)
        leg_angles = mydf3dPostProcess.calculate_leg_angles(save_angles=True)


if __name__ == "__main__":
    fly_dir = os.path.join(NAS2_DIR_JB, "210908_PR_olfac", "Fly1")
    trial_dirs = get_trials_from_fly(fly_dir, startswith="0")[0]
    tmp_process_dir = prepare_for_df3d(trial_dirs=trial_dirs, videos=True, scope=2)
    run_df3d(tmp_process_dir)

