"""
sub-module for pose estimation.
Includes functions to prepare a run-script for DeepFly3D,
run said script and perform post-processing with df3dPostProcessing.
"""
import os
from shutil import copy
import glob
import pickle
import numpy as np
import pandas as pd

from df3dPostProcessing.df3dPostProcessing import df3dPostProcess, df3d_skeleton

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)

from twoppp.utils import makedirs_safe, find_file
from twoppp import TMP_PATH


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

    tmp_process_dir = TMP_PATH if tmp_process_dir is None else tmp_process_dir
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
            images_dir = os.path.join(trial_dir, "images")
            if not os.path.isdir(images_dir):
                images_dir = os.path.join(trial_dir, "behData", "images")
                if not os.path.isdir(images_dir):
                    images_dir = find_file(trial_dir, "images", "images folder")
                    if not os.path.isdir(images_dir):
                        raise FileNotFoundError("Could not find 'images' folder.")
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

def postprocess_df3d_trial(trial_dir, overwrite=False, result_prefix=""):
    """run post-processing of deepfly3d data as defined in the df3dPostProcessing package:
    Align with reference fly template and calculate leg angles.

    Parameters
    ----------
    trial_dir : string
        directory of the trial. should contain an "images" folder at some level of hierarchy
 
    overwrite : bool, optional
        whether to overwrite existing results, by default False
    
    result_prefix: str, optional
        will pre-pend a string to the output files, by default ""
    """
    images_dir = os.path.join(trial_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(trial_dir, "behData", "images")
        if not os.path.isdir(images_dir):
            images_dir = find_file(trial_dir, "images", "images folder")
            if not os.path.isdir(images_dir):
                raise FileNotFoundError("Could not find 'images' folder.")
    df3d_dir = os.path.join(images_dir, "df3d")
    if not os.path.isdir(images_dir):
        df3d_dir = find_file(images_dir, "df3d", "df3d folder")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError("Could not find 'df3d' folder.")

    pose_result = find_file(df3d_dir, name="pose_result*", file_type="pose result file")
    if overwrite or not len(glob.glob(os.path.join(images_dir, "df3d", result_prefix+"joint_angles*"))):
        try:
            mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result, calculate_3d=True,
                                                correct_outliers=True)
        except:
            print("New version of df3d post processing did not work. Will not correct outliers")
            mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result, calculate_3d=True)
        try:
            aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True,
                                                                all_body=True)
        except:
            print("New version of df3d post processing did not work.",
                  "Will not align antennal markers")
            aligned_model = mydf3dPostProcess.align_to_template(scale=True)
        path = pose_result.replace('pose_result',result_prefix+'aligned_pose')
        with open(path, 'wb') as f:
            pickle.dump(aligned_model, f)
        try: 
            leg_angles = mydf3dPostProcess.calculate_leg_angles(save_angles=False)
        except TypeError:
            print("Using old version of df3d post processing!")
            leg_angles = mydf3dPostProcess.calculate_leg_angles()
        path = pose_result.replace('pose_result', result_prefix+'joint_angles')
        with open(path, 'wb') as f:
            pickle.dump(leg_angles, f)

def get_df3d_dataframe(trial_dir, index_df=None, out_dir=None, add_abdomen=True):
    """load pose estimation data into a dataframe, potentially one that is synchronised
    to the two-photon recordings.
    Adds columns for joint position and joint angles.

    Parameters
    ----------
    trial_dir : str
        base directory where pose estimation results can be found

    index_df : pandas Dataframe or str, optional
        pandas dataframe or path of pickle containing dataframe to which the df3d result is added.
        This could, for example, be a dataframe that contains indices for synching with 2p data,
        by default None

    out_dir : str, optional
        if specified, will save the dataframe as .pkl, by default None

    add_abdomen: bool, optional
        if specified, search for abdominal markers in raw pose results

    Returns
    -------
    beh_df: pandas DataFrame
        Dataframe containing behavioural data
    """

    if index_df is not None and isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    if index_df is not None:
        assert isinstance(index_df, pd.DataFrame)
    beh_df = index_df

    images_dir = os.path.join(trial_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(trial_dir, "behData", "images")
        if not os.path.isdir(images_dir):
            images_dir = find_file(trial_dir, "images", "images folder")
            if not os.path.isdir(images_dir):
                raise FileNotFoundError("Could not find 'images' folder.")
    df3d_dir = os.path.join(images_dir, "df3d")
    if not os.path.isdir(images_dir):
        df3d_dir = find_file(images_dir, "df3d", "df3d folder")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError("Could not find 'df3d' folder.")

    # read the angles and convert them into an understandable format
    angles_file = find_file(df3d_dir, name="joint_angles*", file_type="joint angles file")
    with open(angles_file, "rb") as f:
        angles = pickle.load(f)
    leg_keys = []
    _ = [leg_keys.append(key) for key in angles.keys()]
    angle_keys = []
    _ = [angle_keys.append(key) for key in angles[leg_keys[0]].keys()]

    if "Head" in leg_keys:
        N_features = (len(leg_keys) - 1) * len(angle_keys)
    else:
        N_features = len(leg_keys) * len(angle_keys)

    N_samples = len(angles[leg_keys[0]][angle_keys[0]])
    X = np.zeros((N_samples, N_features), dtype="float64")
    X_names = []
    for i_leg, leg in enumerate(leg_keys):
        if leg == "Head":
            continue
        for i_angle, angle in enumerate(angle_keys):
            X[:, i_angle + i_leg*len(angle_keys)] = np.array(angles[leg][angle])
            X_names.append("angle_" + leg + "_" + angle)

    # read the joints from df3d after post-processing and convert them into an understandable format
    joints_file = find_file(df3d_dir, name="aligned_pose*", file_type="aligned pose file")
    with open(joints_file, "rb") as f:
        joints = pickle.load(f)
    leg_keys = list(joints.keys())
    joint_keys = list(joints[leg_keys[0]].keys())
    if "Head" in leg_keys:
        head_keys = list(joints["Head"].keys())
        N_features = (len(leg_keys) - 1) * len(joint_keys) + len(head_keys)
    else:
        head_keys = []
        N_features = len(leg_keys) * len(joint_keys)
    Y = np.zeros((N_samples, N_features*3), dtype="float64")
    Y_names = []
    for i_leg, leg in enumerate(leg_keys):
        if leg == "Head":
            continue
        for i_joint, joint in enumerate(joint_keys):
            Y[:, i_leg*len(joint_keys)*3 + i_joint*3 : i_leg*len(joint_keys)*3 + (i_joint+1)*3] = \
                np.array(joints[leg][joint]["raw_pos_aligned"])
            Y_names += ["joint_" + leg + "_" + joint + i_ax for i_ax in ["_x", "_y", "_z"]]
    if "Head" in leg_keys:
        N_legs = len(leg_keys) - 1
        for i_key, head_key in enumerate(head_keys):
            Y[:, N_legs*len(joint_keys)*3 + i_key*3 : N_legs*len(joint_keys)*3 + (i_key+1)*3] = \
                    np.array(joints["Head"][head_key]["raw_pos_aligned"])
            Y_names += ["joint_Head_" + head_key + i_ax for i_ax in ["_x", "_y", "_z"]]

    if add_abdomen:
        pose_file = find_file(df3d_dir, name="pose_result*", file_type="pose result file")
        with open(pose_file, "rb") as f:
            pose = pickle.load(f)
        abdomen_keys = ["RStripe1", "RStripe2", "RStripe3", "LStripe1", "LStripe2", "LStripe3"]
        abdomen_inds = [df3d_skeleton.index(key) for key in abdomen_keys]

        N_features = len(abdomen_keys)
        Z = np.zeros((N_samples, N_features*3), dtype="float64")
        Z_names = []
        for i_key, (i_abd, abd_key) in enumerate(zip(abdomen_inds, abdomen_keys)):
            Z[:, i_key*3:(i_key+1)*3] = pose["points3d"][:,i_abd, :]
            Z_names += ["joint_Abd_" + abd_key + i_ax for i_ax in ["_x", "_y", "_z"]]

    if beh_df is None:
        # if no index_df was supplied externally,
        # try to get info from trial directory and create empty dataframe
        frames = np.arange(N_samples)
        try:
            fly_dir, trial = os.path.split(trial_dir)
            date_dir, fly = os.path.split(fly_dir)
            _, date_genotype = os.path.split(date_dir)
            date = int(date_genotype[:6])
            genotype = date_genotype[7:]
            fly = int(fly[3:])
            i_trial = int(trial[-3:])
        except:
            date = 123456
            genotype = ""
            fly = -1
            i_trial = -1
        indices = pd.MultiIndex.from_arrays(([date, ] * N_samples,  # e.g 210301
                                                [genotype, ] * N_samples,  # e.g. 'J1xCI9'
                                                [fly, ] * N_samples,  # e.g. 1
                                                [i_trial, ] * N_samples,  # e.g. 1
                                                frames
                                            ),
                                            names=[u'Date', u'Genotype', u'Fly', u'Trial',u'Frame'])
        beh_df = pd.DataFrame(index=indices)

    beh_df[X_names] = X
    beh_df[Y_names] = Y
    if add_abdomen:
        beh_df[Z_names] = Z

    if out_dir is not None:
        beh_df.to_pickle(out_dir)

    return beh_df
