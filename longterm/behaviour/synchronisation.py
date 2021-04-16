import sys, os
import numpy as np

import utils2p
import utils2p.synchronization

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm import load

def get_frame_times_indices(trial_dir, crop_2p_start_end=0):
    sync_file = utils2p.find_sync_file(trial_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)
    processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])
    beh_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), processed_lines["Cameras"], processed_lines["Frame Counter"])
    # crop start and end of 2p, for example in case of denoising, which causes 30 frames at start and end to be lost
    frame_times_2p = frame_times_2p[crop_2p_start_end:-crop_2p_start_end] if crop_2p_start_end != 0 else frame_times_2p
    beh_frame_idx -= crop_2p_start_end
    beh_frame_idx[beh_frame_idx<0] = -9223372036854775808  # smallest possible uint64 number
    beh_frame_idx[beh_frame_idx>len(frame_times_2p)] = -9223372036854775808
    return frame_times_2p, frame_times_beh, beh_frame_idx

if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")  # 210216_J1xCI9 fly1 trial 0
    fly_dirs = load.get_flies_from_datedir(date_dir)
    trial_dirs = load.get_trials_from_fly(fly_dirs)
    trial_dir = trial_dirs[0][0]
    times_2p, times_beh, index = get_frame_times_indices(trial_dir, crop_2p_start_end=30)
    pass
