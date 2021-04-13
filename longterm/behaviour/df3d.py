import sys, os
from shutil import copy
import glob
import pickle

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import makedirs_safe, find_file

from df3dPostProcessing.df3dPostProcessing import df3dPostProcess

def prepare_for_df3d(trial_dirs, videos=True, scope=2, tmp_process_dir=None, overwrite=False):
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
    run_script = os.path.join(tmp_process_dir, "run_df3d.sh")
    folders_dir = os.path.join(tmp_process_dir, "folders.txt")
    if not os.path.isfile(run_script) or not os.path.isfile(folders_dir):
        raise FileNotFoundError
    if os.stat(folders_dir).st_size:  # confirm that the folders.txt file is not empty
        os.chdir(tmp_process_dir)
        os.system("pwd")
        os.system("sh run_df3d.sh")

def postprocess_df3d_trial(trial_dir, overwrite=False):
    images_dir = find_file(trial_dir, "images", "images folder")
    pose_result = glob.glob(os.path.join(images_dir, "df3d", "pose_result*"))[0]
    if overwrite or not len(glob.glob(os.path.join(images_dir, "df3d", "joint_angles*"))):
        mydf3dPostProcess = df3dPostProcess(exp_dir=pose_result)
        aligned_model = mydf3dPostProcess.align_to_template(interpolate=False, scale=True)
        path = pose_result.replace('pose_result','aligned_pose')
        with open(path, 'wb') as f:
            pickle.dump(aligned_model, f)
        leg_angles = mydf3dPostProcess.calculate_leg_angles(save_angles=True)


if __name__ == "__main__":
    trial_dirs = ["/mnt/NAS/JB/210301_J1xCI9/Fly1/001_xz"]
    tmp_process_dir = prepare_for_df3d(trial_dirs=trial_dirs, videos=True, scope=2)
    run_df3d(tmp_process_dir)



    






    