# cv2.putText(
#                 img, line, (pos[0], pos[1] + j * 40), font, scale, color, line_type
#             )

# Jonas Braun
# 03.06.2021

import os
import glob
import cv2
import numpy as np
import sys
import json
from tqdm import tqdm
import pickle

import utils2p
from utils2p.synchronization import get_lines_from_h5_file, process_cam_line, process_stimulus_line, crop_lines, get_times, SyncMetadata

VIDEO_NAME = ["camera_", ".mp4"]
FRAME_NAME = ["camera_","_img_", ".jpg"]


def mp4fromframes(base_dir):
    pos=(10, 150)
    font=cv2.FONT_HERSHEY_DUPLEX
    scale=3
    color=(255, 255, 255)
    line_type=2

    with open(os.path.join(base_dir, "capture_metadata.json")) as f:
        metadata = json.load(f)
        frame_rate = metadata['FPS']
        N_cams = metadata['Number of Cameras']
        N_frames = metadata['Number of Frames']['0']
        width = metadata['ROI']['Width']['0']
        height = metadata['ROI']['Height']['0']

    for i_cam in tqdm(np.arange(N_cams)):
        if i_cam != 0:
            THIS_VIDEO = os.path.join(base_dir, VIDEO_NAME[0]+str(i_cam)+VIDEO_NAME[1])
            FIRST_FRAME = os.path.join(base_dir, FRAME_NAME[0]+str(i_cam)+FRAME_NAME[1]+str(0)+FRAME_NAME[2])
            if not os.path.isfile(THIS_VIDEO) or os.path.isfile(FIRST_FRAME):
                # make video from frames
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(THIS_VIDEO, fourcc, frame_rate, (width,height))

                for i_frame in tqdm(np.arange(N_frames)):
                    img = cv2.imread(os.path.join(base_dir, FRAME_NAME[0]+str(i_cam)+FRAME_NAME[1]+str(i_frame)+FRAME_NAME[2]))
                    cv2.putText(
                        img, "{:5d}".format(i_frame), (pos[0], pos[1]), font, scale, color, line_type
                    )
                    video.write(img)

                cv2.destroyAllWindows()
                video.release()
        else:
            # add text to existing video
            pos=(10, 50)
            scale = 2
            THIS_VIDEO = os.path.join(base_dir, VIDEO_NAME[0]+str(i_cam)+VIDEO_NAME[1])
            cap = cv2.VideoCapture(THIS_VIDEO)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(THIS_VIDEO[:-4]+"_frame.mp4", fourcc, frame_rate, (width,height))
            for i_frame in tqdm(range(min(N_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))):
                ret, img = cap.read()
                cv2.putText(
                    img, "{:5d}".format(i_frame), (pos[0], pos[1]), font, scale, color, line_type
                )
                video.write(img)

            cv2.destroyAllWindows()
            video.release()
            

        # delete all frames that were used to create the video
        for img in glob.glob(os.path.join(base_dir, FRAME_NAME[0]+str(i_cam)+FRAME_NAME[1]+"*"+FRAME_NAME[2])):
            try: 
                os.remove(img)
            except(FileNotFoundError):
                pass

def process_lines(sync_file, sync_metadata_file, seven_camera_metadata_file=None):
    """
    This function extracts all the standard lines and processes them.
    It works for both microscopes.
    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    sync_metadata_file : str
        Path to the synchronization metadata file.
    metadata_2p_file : str
        Path to the ThorImage metadata file.
    seven_camera_metadata_file : str
        Path to the metadata file of the 7 camera system.
    Returns
    -------
    processed_lines : dictionary
        Dictionary with all processed lines.
    
    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> experiment_dir = "data/mouse_kidney_raw/"
    >>> sync_file = utils2p.find_sync_file(experiment_dir)
    >>> metadata_file = utils2p.find_metadata_file(experiment_dir)
    >>> sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
    >>> seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
    >>> processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    """
    processed_lines = {}
    try:
        processed_lines["odor"], processed_lines["pid"] = get_lines_from_h5_file(sync_file, ["odor", "pid",])
    except:
        processed_lines["odor"], processed_lines["Cameras"] = get_lines_from_h5_file(sync_file, ["odor", "Cameras",])
        processed_lines["Cameras"] = process_cam_line(processed_lines["Cameras"], seven_camera_metadata_file)
    # processed_lines["odor"] = process_stimulus_line(processed_lines["odor"])
    
    # Get times of ThorSync ticks
    metadata = SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = get_times(len(processed_lines["odor"]), freq)
    processed_lines["Times"] = times

    return processed_lines


if __name__=="__main__":
    MAKE_VIDEOS = False
    GET_SYNC_DATA = False
    GET_PID_DATA = True

    base_dir = "/mnt/labserver/BRAUN_Jonas/Experimental_data/Olfactometer/210603_olfactory_test"
    trials = ["001_100_left", "002_50_left", "003_30_left"]
    trial_dirs = [os.path.join(base_dir, trial) for trial in trials]
    img_dirs = [os.path.join(trial_dir, "behData","images") for trial_dir in trial_dirs]
    sync_names = ["sync001", "sync002", "sync003"]
    sync_dirs = [os.path.join(base_dir, sync_name) for sync_name in sync_names]
    sync_out_files = [os.path.join(base_dir, "sync_out_00{}.pkl".format(i+1)) for i in range(3)]

    if MAKE_VIDEOS:
        for img_dir in img_dirs:
            print(img_dir)
            mp4fromframes(img_dir)

    elif GET_SYNC_DATA:
        for i_trial, (sync_dir, img_dir, sync_out_file) in enumerate(zip(sync_dirs, img_dirs, sync_out_files)):
            print(sync_dir)
            sync_file = utils2p.find_sync_file(sync_dir)
            sync_metadata_file = utils2p.find_sync_metadata_file(sync_dir)
            seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(img_dir)
            processed_lines = process_lines(sync_file, sync_metadata_file, seven_camera_metadata_file)
            with open(sync_out_file, "wb") as f:
                pickle.dump(processed_lines, f)
    elif GET_PID_DATA:
        # base_dir = "/mnt/labserver/BRAUN_Jonas/Experimental_data/Olfactometer/210615_olfactory_pid_test"
        base_dir = "/mnt/NAS2/JB/210826_olfac_pid"  
        # 210726_olfac_pid # 210720_olfac_pid  # 210721_olfac_pid  # 210722_olfac_pid
        # sync_names = ["10_mL", "100_mL"]
        sync_names = ["sync001"]  # , "sync002", "sync003", "sync004", "sync005", 
                      # "sync006", "sync007", "sync008", "sync009", "sync010",
                      # "sync011", "sync012", "sync013", "sync014"]
        sync_dirs = [os.path.join(base_dir, sync_name) for sync_name in sync_names]
        sync_out_files = [os.path.join(base_dir, sync_name+".pkl") for sync_name in sync_names]
        for i_trial, (sync_dir, sync_out_file) in enumerate(zip(sync_dirs, sync_out_files)):
            print(sync_dir)
            sync_file = utils2p.find_sync_file(sync_dir)
            sync_metadata_file = utils2p.find_sync_metadata_file(sync_dir)
            processed_lines = process_lines(sync_file, sync_metadata_file)
            with open(sync_out_file, "wb") as f:
                pickle.dump(processed_lines, f)
            