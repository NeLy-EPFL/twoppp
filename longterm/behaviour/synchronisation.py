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
from longterm.utils.df import get_multi_index_trial_df

def get_frame_times_indices(trial_dir, crop_2p_start_end=0, beh_trial_dir=None, sync_trial_dir=None, opflow=False):
    beh_trial_dir = trial_dir if beh_trial_dir is None else beh_trial_dir
    sync_trial_dir = trial_dir if sync_trial_dir is None else sync_trial_dir

    sync_file = utils2p.find_sync_file(sync_trial_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_trial_dir)
    processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, 
                                                              metadata_file, seven_camera_metadata_file)
    frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], 
                                                             processed_lines["Times"])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], 
                                                              processed_lines["Times"])
    beh_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), 
                                                              processed_lines["Cameras"], 
                                                              processed_lines["Frame Counter"])
    # crop start and end of 2p, for example in case of denoising, which causes 30 frames at start and end to be lost
    frame_times_2p = frame_times_2p[crop_2p_start_end:-crop_2p_start_end] \
                     if crop_2p_start_end != 0 else frame_times_2p
    beh_frame_idx -= crop_2p_start_end
    beh_frame_idx[beh_frame_idx<0] = -9223372036854775808  # smallest possible uint64 number
    beh_frame_idx[beh_frame_idx>=len(frame_times_2p)] = -9223372036854775808
    if opflow:
        frame_times_opflow = utils2p.synchronization.get_start_times(processed_lines["Optical flow"], 
                                                                    processed_lines["Times"])
        opflow_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_opflow)), 
                                                                    processed_lines["Optical flow"], 
                                                                    processed_lines["Frame Counter"])
        opflow_frame_idx -= crop_2p_start_end
        opflow_frame_idx[opflow_frame_idx<0] = -9223372036854775808  # smallest possible uint64 number
        opflow_frame_idx[opflow_frame_idx>=len(frame_times_2p)] = -9223372036854775808
        return frame_times_2p, frame_times_beh, beh_frame_idx, frame_times_opflow, opflow_frame_idx
    else:
        return frame_times_2p, frame_times_beh, beh_frame_idx

def get_synchronised_trial_dataframes(trial_dir, crop_2p_start_end=0, beh_trial_dir=None, sync_trial_dir=None, trial_info=None, 
                                      opflow=False, df3d=True, opflow_out_dir=None, df3d_out_dir=None, twop_out_dir=None):
    if opflow:
        frame_times_2p, frame_times_beh, beh_frame_idx, frame_times_opflow, opflow_frame_idx = \
            get_frame_times_indices(trial_dir, crop_2p_start_end=crop_2p_start_end, 
                                    beh_trial_dir=beh_trial_dir, sync_trial_dir=sync_trial_dir, opflow=True)

    else:
        frame_times_2p, frame_times_beh, beh_frame_idx = \
            get_frame_times_indices(trial_dir, crop_2p_start_end=crop_2p_start_end, 
                                    beh_trial_dir=beh_trial_dir, sync_trial_dir=sync_trial_dir, opflow=False)
        frame_times_opflow = None
        opflow_frame_idx = None
    
    twop_df = get_multi_index_trial_df(trial_info, len(frame_times_2p), t=frame_times_2p)
    if twop_out_dir is not None:
        twop_df.to_pickle(twop_out_dir)
    df3d_df = get_multi_index_trial_df(trial_info, len(frame_times_beh), t=frame_times_beh, 
                                 twop_index=beh_frame_idx) if df3d else None
    if df3d_out_dir is not None:
        df3d_df.to_pickle(df3d_out_dir)
    opflow_df = get_multi_index_trial_df(trial_info, len(frame_times_opflow), t=frame_times_opflow, 
                                   twop_index=opflow_frame_idx) if opflow else None
    if opflow_out_dir is not None:
        opflow_df.to_pickle(opflow_out_dir)
    return twop_df, df3d_df, opflow_df
    
    # these two functions are just wrappers around the numpy functions to apply them across dimension 0 only
def reduce_mean(values):
    return np.mean(values, axis=0)
def reduce_std(values):
    return np.std(values, axis=0)
def reduce_behaviour(values):
    unique_values, N_per_unique = np.unique(values, return_counts=True)
    i_max = np.argmax(N_per_unique)
    if N_per_unique[i_max] < 0.75 * len(values):
        return ""
    else:
        return unique_values[i_max]

def reduce_during_2p_frame(twop_index, values, function=reduce_mean):
    """
    Reduces all values occuring during the acquisition of a
    two-photon imaging frame to a single value using the `function` given by the user.
    Parameters
    ----------
    twop_index : numpy array
        1d array holding frame indices of one trial.
    values : numpy array
        Values upsampled to the frequency of ThorSync,
        i.e. 1D numpy array of the same length as
        `frame_counter` or 2D numpy array of the same length.
    function : function
        Function used to reduce the value,
        e.g. np.mean for 1D variables
    Returns
    -------
    reduced : numpy array
        Numpy array with value for each two-photon imaging frame.
    """
    
    if len(twop_index) != len(values):
        raise ValueError("twop_index and values need to have the same length.")
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=1)
        squeeze = True
    else:
        squeeze = False
    N_samples, N_variables = values.shape
    
    index_unique = np.unique(twop_index)
    index_unique = np.delete(index_unique, index_unique==-9223372036854775808)
    
    dtype = values.dtype
    if np.issubdtype(dtype, np.number):
        dtype = np.float
    else:
        dtype = np.object
    reduced = np.empty((len(index_unique), N_variables), dtype=dtype)

    for i, index in enumerate(index_unique):
        reduced[i] = function(values[twop_index==index, :])

    return np.squeeze(reduced) if squeeze else reduced

if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")  # 210216_J1xCI9 fly1 trial 0
    fly_dirs = load.get_flies_from_datedir(date_dir)
    trial_dirs = load.get_trials_from_fly(fly_dirs)
    trial_dir = trial_dirs[0][0]
    times_2p, times_beh, index = get_frame_times_indices(trial_dir, crop_2p_start_end=30)
    pass