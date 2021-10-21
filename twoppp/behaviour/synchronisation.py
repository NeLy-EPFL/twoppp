"""
Sub-module to synchronise between two photon, behaviour & ball tracking data.
Relies heavlily on utils2p.synchronisation:
https://github.com/NeLy-EPFL/utils2p/blob/master/utils2p/synchronization.py
"""

import sys
import os
import numpy as np

import utils2p
import utils2p.synchronization

from twoppp import load
from twoppp.utils.df import get_multi_index_trial_df

def get_frame_times_indices(trial_dir, crop_2p_start_end=0, beh_trial_dir=None,
                            sync_trial_dir=None, opflow=False):
    """get the times of different data acquisition modalities from the sync files.
    Also computes the frame indices to lateron sync between behavioural and two photon data.

    Attention: the absolute time is only precise to the second because it is based on the 
    unix time stamp from the metadatafile of the recording.

    Parameters
    ----------
    trial_dir : str
        directory containing the 2p data and ThorImage output

    crop_2p_start_end : int, optional
        specify if DeepInterpolation was used, because it crops 30 frames in the front
        and 30 frames in the back., by default 0

    beh_trial_dir : [type], optional
        directory containing the 7 camera data. If not specified, will be set equal
        to trial_dir,, by default None

    sync_trial_dir : [type], optional
        directory containing the output of ThorSync. If not specified, will be set equal
        to trial_dir, by default None

    opflow : bool, optional
        whether to load optical flow. This will also change the output format
        (5 instead of 3 outputs), by default False

    Returns
    -------
    unix_t_start: int
        start time of the experiment as a unix time stamp

    frame_times_2p: numpy array
        absolute times when the two photon frames were acquired

    frame_times_beh: numpy array
        absolute times when the 7 camera images were acquired

    beh_frame_idx: numpy array
        same length as frame_times_beh. contains the index of the two photon frame
        that was acquired during each behaviour frame

    (frame_times_opflow): numpy array
        only returned if opflow == True
        absolute times when the optic flow sensor values were acquired

    (opflow_frame_idx): numpy array
        only returned if opflow == True
        same length as frame_times_opflow. contains the index of the two photon frame
        that was acquired during each optical flow sample
    """
    beh_trial_dir = trial_dir if beh_trial_dir is None else beh_trial_dir
    sync_trial_dir = trial_dir if sync_trial_dir is None else sync_trial_dir

    sync_file = utils2p.find_sync_file(sync_trial_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_trial_dir)

    unix_t_start = int(utils2p.Metadata(metadata_file).get_metadata_value("Date", "uTime"))

    processed_lines = utils2p.synchronization.get_processed_lines(sync_file, sync_metadata_file,
                                                              metadata_file,
                                                              seven_camera_metadata_file)
    frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"],
                                                             processed_lines["Times"])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"],
                                                              processed_lines["Times"])
    beh_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)),
                                                              processed_lines["Cameras"],
                                                              processed_lines["Frame Counter"])
    # crop start and end of 2p, for example in case of denoising,
    # which causes 30 frames at start and end to be lost
    frame_times_2p = frame_times_2p[crop_2p_start_end:-crop_2p_start_end] \
                     if crop_2p_start_end != 0 else frame_times_2p
    beh_frame_idx -= crop_2p_start_end
    beh_frame_idx[beh_frame_idx<0] = -9223372036854775808  # smallest possible uint64 number
    beh_frame_idx[beh_frame_idx>=len(frame_times_2p)] = -9223372036854775808
    if opflow:
        frame_times_opflow = utils2p.synchronization.get_start_times(
            processed_lines["Optical flow"],
            processed_lines["Times"]
        )
        opflow_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(
            np.arange(len(frame_times_opflow)),
            processed_lines["Optical flow"],
            processed_lines["Frame Counter"]
        )
        opflow_frame_idx -= crop_2p_start_end
        opflow_frame_idx[opflow_frame_idx<0] = -9223372036854775808  # smallest possible uint64
        opflow_frame_idx[opflow_frame_idx>=len(frame_times_2p)] = -9223372036854775808
        return unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx, frame_times_opflow, opflow_frame_idx
    else:
        return unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx

def get_synchronised_trial_dataframes(trial_dir, crop_2p_start_end=0, beh_trial_dir=None,
                                      sync_trial_dir=None, trial_info=None,
                                      opflow=False, df3d=True, opflow_out_dir=None,
                                      df3d_out_dir=None, twop_out_dir=None):
    """get trial dataframes that include the time and synchronisation variables.
    Important: they do not contain any data yet, but only the timing information
    used to synchronise different variables

    Parameters
    ----------
    trial_dir : str
        directory containing the 2p data and ThorImage output

    crop_2p_start_end : int, optional
        specify if DeepInterpolation was used, because it crops 30 frames in the front
        and 30 frames in the back., by default 0

    beh_trial_dir : str, optional
        directory containing the 7 camera data. If not specified, will be set equal
        to trial_dir, by default None

    sync_trial_dir : str, optional
        directory containing the output of ThorSync. If not specified, will be set equal
        to trial_dir, by default None

    trial_info : dict, optional
        information about trial used to generate MultiIndex DataFrame, by default None

    opflow : bool, optional
        whether to create a dataframe for optic flow as well, by default False

    df3d : bool, optional
        whether to create a dataframe for 7cam data as well, by default True

    opflow_out_dir : str, optional
        absolute file path where to store the optical flow DataFrame, by default None

    df3d_out_dir : str, optional
        absolute file path where to store the 7cam DataFrame, by default None

    twop_out_dir : str, optional
        absolute file path where to store the twop data DataFrame, by default None

    Returns
    -------
    twop_df: pandas DataFrame
        two-photon data DataFrame

    df3d_df: pandas DataFrame or None
        7 camera data DataFrame

    opflow_df: pandas DataFrame or None
        optical flow data DataFrame
    """
    if opflow:
        unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx, \
            frame_times_opflow, opflow_frame_idx = \
            get_frame_times_indices(trial_dir, crop_2p_start_end=crop_2p_start_end,
                                    beh_trial_dir=beh_trial_dir, sync_trial_dir=sync_trial_dir,
                                    opflow=True)

    else:
        unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx = \
            get_frame_times_indices(trial_dir, crop_2p_start_end=crop_2p_start_end,
                                    beh_trial_dir=beh_trial_dir, sync_trial_dir=sync_trial_dir,
                                    opflow=False)
        frame_times_opflow = None
        opflow_frame_idx = None

    twop_df = get_multi_index_trial_df(trial_info, len(frame_times_2p), t=frame_times_2p,
                                       abs_t_start=unix_t_start)
    if twop_out_dir is not None:
        twop_df.to_pickle(twop_out_dir)
    df3d_df = get_multi_index_trial_df(
        trial_info, len(frame_times_beh), t=frame_times_beh,
        twop_index=beh_frame_idx, abs_t_start=unix_t_start) if df3d else None
    if df3d_out_dir is not None:
        df3d_df.to_pickle(df3d_out_dir)
    opflow_df = get_multi_index_trial_df(
        trial_info, len(frame_times_opflow), t=frame_times_opflow,
        twop_index=opflow_frame_idx, abs_t_start=unix_t_start) if opflow else None
    if opflow_out_dir is not None and opflow_df is not None:
        opflow_df.to_pickle(opflow_out_dir)
    return twop_df, df3d_df, opflow_df

    # these functions are just wrappers around the numpy functions to apply them across dim 0 only
def reduce_mean(values):
    return np.mean(values, axis=0)
def reduce_std(values):
    return np.std(values, axis=0)
def reduce_min(values):
    return np.min(values, axis=0)
def reduce_max(values):
    return np.max(values, axis=0)
def reduce_bin_p(values, thres):
    return np.mean(values, axis=0) > thres
def reduce_bin_50p(values):
    return reduce_bin_p(values, 0.5)
def reduce_bin_75p(values):
    return reduce_bin_p(values, 0.75)
def reduce_bin_90p(values):
    return reduce_bin_p(values, 0.9)

def reduce_behaviour(values, thres=0.75, default=""):
    unique_values, N_per_unique = np.unique(values, return_counts=True)
    i_max = np.argmax(N_per_unique)
    if N_per_unique[i_max] < thres * len(values):
        return default
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
