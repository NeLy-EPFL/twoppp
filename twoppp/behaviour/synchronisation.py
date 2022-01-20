"""
Sub-module to synchronise between two photon, behaviour & ball tracking data.
Relies heavlily on utils2p.synchronisation:
https://github.com/NeLy-EPFL/utils2p/blob/master/utils2p/synchronization.py
"""

import sys
import os
import numpy as np
import json

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
    try:
        metadata_file = utils2p.find_metadata_file(trial_dir)
        twop_present = True
    except:
        metadata_file = None
        twop_present = False
        print(f"Could not finde 2p data for trial {trial_dir}. will proceed with behavioural data only.")
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_trial_dir)

    unix_t_start = int(utils2p.Metadata(metadata_file).get_metadata_value("Date", "uTime")) if twop_present else 0
    # don't use utils2p.synchronization, but temporarily use the one below to keep flexibility if twop was not recorded
    processed_lines = get_processed_lines(sync_file, sync_metadata_file,
                                                              metadata_file,
                                                              seven_camera_metadata_file)
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"],
                                                              processed_lines["Times"])
    if twop_present:
        frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"],
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
    else:
        frame_times_2p = frame_times_beh
        beh_frame_idx = np.ones_like(frame_times_beh) * -9223372036854775808
    if opflow:
        frame_times_opflow = utils2p.synchronization.get_start_times(
            processed_lines["Optical flow"],
            processed_lines["Times"]
        )
        if twop_present:
            opflow_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(
                np.arange(len(frame_times_opflow)),
                processed_lines["Optical flow"],
                processed_lines["Frame Counter"]
            )
            opflow_frame_idx -= crop_2p_start_end
            opflow_frame_idx[opflow_frame_idx<0] = -9223372036854775808  # smallest possible uint64
            opflow_frame_idx[opflow_frame_idx>=len(frame_times_2p)] = -9223372036854775808
        else:
            opflow_frame_idx = np.ones_like(frame_times_opflow) * -9223372036854775808
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

def reduce_groups(group_index, values, function=reduce_mean):
    """
    Reduces all values/arrays belonging to one group to a single value/array.
    A group can be consecutive samples during an event at lower frame rate
    or just a random subset of samples in 'values'.
    Supply -9223372036854775808 for a sample that does not belong to any group.

    Parameters
    ----------
    group_index : numpy array
        1d array holding frame indices of one trial.

    values : numpy array
        Values upsampled to the frequency of ThorSync,
        i.e. 1D numpy array of the same length as
        `group_index` or ND numpy array of the same length.

    function : [function, optional
        Function used to reduce the value,
        e.g. np.mean for 1D variables, default is np.mean(values, axis=0)

    Returns
    -------
    reduced : numpy array
        Numpy array with value/array for each low frequency frame.
    """

    return reduce_during_2p_frame(twop_index=group_index, values=values, function=function)

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
        `twop_index` or ND numpy array of the same length.

    function : function, optional
        Function used to reduce the value,
        e.g. np.mean for 1D variables, default is np.mean(values, axis=0)

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
    N_samples = values.shape[0]

    index_unique = np.unique(twop_index)
    index_unique = np.delete(index_unique, index_unique==-9223372036854775808)

    dtype = values.dtype
    if np.issubdtype(dtype, np.number):
        dtype = np.float
    else:
        dtype = np.object
    reduced = np.empty((len(index_unique),) + values.shape[1:], dtype=dtype)

    for i, index in enumerate(index_unique):
        reduced[i] = function(values[twop_index==index, :])

    return np.squeeze(reduced) if squeeze else reduced


def get_processed_lines(sync_file,
                        sync_metadata_file,
                        metadata_2p_file,
                        seven_camera_metadata_file=None):
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
    >>> processed_lines = utils2p.synchronization.get_processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    """
    processed_lines = {}
    processed_lines["Capture On"], processed_lines[
        "Frame Counter"] = utils2p.synchronization.get_lines_from_sync_file(
            sync_file, ["Capture On", "Frame Counter"])

    try:
        # For microscope 1
        processed_lines["CO2"], processed_lines["Cameras"], processed_lines[
            "Optical flow"] = utils2p.synchronization.get_lines_from_sync_file(sync_file, [
                "CO2_Stim",
                "Basler",
                "OpFlow",
            ])
    except KeyError:
        # For microscope 2
        processed_lines["CO2"], processed_lines[
            "Cameras"] = utils2p.synchronization.get_lines_from_h5_file(sync_file, [
                "CO2",
                "Cameras",
            ])

    processed_lines["Cameras"] = utils2p.synchronization.process_cam_line(processed_lines["Cameras"],  # TODO
                                                  seven_camera_metadata_file)
    if metadata_2p_file is not None:
        metadata_2p = main.Metadata(metadata_2p_file)
        processed_lines["Frame Counter"] = utils2p.synchronization.process_frame_counter(
            processed_lines["Frame Counter"], metadata_2p)

    processed_lines["CO2"] = utils2p.synchronization.process_stimulus_line(processed_lines["CO2"])

    if "Optical flow" in processed_lines.keys():
        processed_lines["Optical flow"] = utils2p.synchronization.process_optical_flow_line(
            processed_lines["Optical flow"])

    if metadata_2p_file is not None:
        mask = np.logical_and(processed_lines["Capture On"],
                            processed_lines["Frame Counter"] >= 0)

        # Make sure the clipping start just before the
        # acquisition of the first frame
        indices = np.where(mask)[0]
        mask[max(0, indices[0] - 1)] = True

        for line_name, _ in processed_lines.items():
            processed_lines[line_name] = utils2p.synchronization.crop_lines(mask, [
                processed_lines[line_name],
            ])[0]

    # Get times of ThorSync ticks
    metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = utils2p.synchronization.get_times(len(processed_lines["Frame Counter"]), freq)
    processed_lines["Times"] = times

    return processed_lines

def process_cam_line(line, seven_camera_metadata, align_if_missing="end"):
    """
    DO NOT USE! WORK IN PROGRESS!!!
    
    Removes superfluous signals and uses frame numbers in array.
    The cam line signal form the h5 file is a binary sequence.
    Rising edges mark the acquisition of a new frame.
    The setup keeps producing rising edges after the acquisition of the
    last frame. These rising edges are ignored.
    This function converts the binary line to frame numbers using the
    information stored in the metadata file of the seven camera setup.
    In the metadata file the keys are the indices of the file names
    and the values are the grabbed frame numbers. Suppose the 3
    frame was dropped. Then the entries in the dictionary will
    be as follows:
    "2": 2
    "3": 4
    "4": 5

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    seven_camera_metadata : string
        Path to the json file saved by our camera software.
        This file is usually located in the same folder as the frames
        and is called 'capture_metadata.json'. If None, it is assumed
        that no frames were dropped.

    Returns
    -------
    processed_line : numpy array
        Array with frame number for each time point.
        If no frame is available for a given time,
        the value is -9223372036854775808.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> seven_camera_metadata = utils2p.find_seven_camera_metadata_file("data/mouse_kidney_raw")
    >>> line_names = ["Basler"]
    >>> (cam_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> set(np.diff(cam_line))
    {0, 8, 4294967288}
    >>> processed_cam_line = utils2p.synchronization.process_cam_line(cam_line, seven_camera_metadata)
    >>> set(np.diff(processed_cam_line))
    {0, 1, -9223372036854775808, 9223372036854775749}
    >>> cam_line = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    >>> utils2p.synchronization.process_cam_line(cam_line, seven_camera_metadata=None)
    array([-9223372036854775808,                    0,                    0,
                              0,                    0,                    0,
                              1,                    1,                    1,
                              1,                    1])
    """
    # Check that sequence is binary
    if len(set(line)) > 2:
        raise ValueError("Invalid line argument. Sequence is not binary.")

    # Find indices of the start of each frame acquisition
    rising_edges = utils2p.synchronization.edges(line, (0, np.inf))[0]

    # Load capture metadata or generate default
    if seven_camera_metadata is not None:
        with open(seven_camera_metadata, "r") as f:
            capture_info = json.load(f)
    else:
        capture_info = utils2p.synchronization._capture_metadata([
            len(rising_edges),
        ])

    # Find the number of frames for each camera
    n_frames = []
    for cam_idx in capture_info["Frame Counts"].keys():
        max_in_json = max(capture_info["Frame Counts"][cam_idx].values())
        n_frames.append(max_in_json + 1)

    # Ensure all cameras acquired the same number of frames
    if len(np.unique(n_frames)) > 1:
        raise utils2p.synchronization.SynchronizationError(
            "The frames across cameras are not synchronized.")

    # Last rising edge that corresponds to a frame
    last_tick = max(n_frames)

    # check that there is a rising edge for every frame
    if len(rising_edges) < last_tick and not align_if_missing:
        raise ValueError(
            "The provided cam line and metadata are inconsistent. " +
            "cam line has less frame acquisitions than metadata.")
    elif len(rising_edges) < last_tick and align_if_missing == "end":
        rising_edges = rising_edges[-last_tick:]
        print("The provided cam line and metadata are inconsistent. " +
            f"cam line has {last_tick-len(rising_edges)}less frame acquisitions than metadata." +
            "Will align recording to the end")
    elif len(rising_edges) < last_tick and align_if_missing == "start":
        rising_edges = rising_edges[:last_tick]
        print("The provided cam line and metadata are inconsistent. " +
            f"cam line has {last_tick-len(rising_edges)}less frame acquisitions than metadata." +
            "Will align recording to the start")

    # Ensure correct handling if no rising edges are present after last frame
    if len(rising_edges) == int(last_tick):
        average_frame_length = int(np.mean(np.diff(rising_edges)))
        last_rising_edge = rising_edges[-1]
        additional_edge = last_rising_edge + average_frame_length
        if additional_edge > len(line):
            additional_edge = len(line)
        rising_edges = list(rising_edges)
        rising_edges.append(additional_edge)
        rising_edges = np.array(rising_edges)

    processed_line = np.ones_like(line) * np.nan

    current_frame = 0
    first_camera_used = sorted(list(capture_info["Frame Counts"].keys()))[0]
    for i, (start, stop) in enumerate(
            zip(rising_edges[:last_tick], rising_edges[1:last_tick + 1])):
        if capture_info["Frame Counts"][first_camera_used][str(current_frame +
                                                               1)] <= i:
            current_frame += 1
        processed_line[start:stop] = current_frame
    return processed_line.astype(np.int)