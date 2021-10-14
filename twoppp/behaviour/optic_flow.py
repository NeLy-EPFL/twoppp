"""
sub-module to analyse data from the optical flow sensors
"""
import os
import sys
import numpy as np
import pandas as pd

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
TWOPPP_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
sys.path.append(MODULE_PATH)

from twoppp.utils import find_file
from twoppp.behaviour.synchronisation import reduce_during_2p_frame, reduce_min

gain0X = round(1/1.52,2)
gain0Y = round(1/1.48,2)
gain1X = round(1/1.48,2)
gain1Y = round(1/1.44,2)

gains = {"gain0X": gain0X,
    "gain0Y": gain0Y,
    "gain1X": gain1X,
    "gain1Y": gain1Y,
    }

def load_opflow(beh_trial_dir):
    """
    This function opens the optic flow measurements values initially stored in a text file.

    Parameters
    ----------
    beh_trial_dir : string
        absolute path of the folder where the optic flow file "OptFlow.txt" is located

    Returns
    -------
    pandas DataFrame
        raw sensor values of the optic flow sensor
    """
    opflow_file = find_file(beh_trial_dir, 'OptFlow.txt')
    cols = ['sens0X','sens0Y','sens1X','sens1Y','OpflowTime']
    flow_data = pd.read_table(opflow_file, sep=',', header=None, names=cols)
    return flow_data

def calibrate_opflow(flow_data, gains=gains):
    """
    This function returns the sensor values times the calibration gain.

    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing the raw sensor values as outputted by load_opflow()

    Returns
    -------
    pandas Dataframe
        same dataframe, but values multiplied with calibration gain
    """
    flow_data["sens0X"] *= gains["gain0X"]
    flow_data["sens0Y"] *= gains["gain0Y"]
    flow_data["sens1X"] *= gains["gain1X"]
    flow_data["sens1Y"] *= gains["gain1Y"]
    return flow_data

def compute_velocities(flow_data):
    """
    This function computes the AP, ML and Yaw rot/s from the sensor measurements.
    Equations from: https://www.nature.com/articles/nmeth.1468
    
    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing optical flow sensor values

    Returns
    -------
    pandas Dataframe
        same dataframe with additional fields velForw, velSide, velTurn
    """
    velForw = -((flow_data["sens0Y"] + flow_data["sens1Y"]) * np.cos(np.deg2rad(45)))
    velSide = (flow_data["sens0Y"] - flow_data["sens1Y"]) * np.sin(np.deg2rad(45))
    velTurn = (flow_data["sens0X"] + flow_data["sens1X"]) / float(2)

    flow_data["velForw"] = velForw
    flow_data["velSide"] = velSide
    flow_data["velTurn"] = velTurn
    return flow_data

def opflow_filter(x, winsize=80):
    return np.convolve(x, np.ones((winsize))/winsize, mode="same")

def filter_velocities(flow_data, winsize=80):
    """apply moving averag filter to optical flow data inside the same pandas DataFrame

    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing velocities computed from optical flow, e.g. output of compute_velocities()
    winsize : int, optional
        window size, by default 80

    Returns
    -------
    pandas DataFrame
        same DataFrame with velocitie fields smoothed
    """
    flow_data["velForw"] = opflow_filter(flow_data["velForw"], winsize)
    flow_data["velSide"] = opflow_filter(flow_data["velSide"], winsize)
    flow_data["velTurn"] = opflow_filter(flow_data["velTurn"], winsize)
    return flow_data

def forward_walking(flow_data, thres_walk=0.03, winsize=100): 
    """apply a threshold to check whether the fly is forward walking or not.
    adds field "walk" to the dataframe.

    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing velocities computed from optical flow, e.g. output of compute_velocities()
        Ideally already filtered using filter_velocities()
    thres_walk : float, optional
        threshold to apply on forward walking units (rotations/s), by default 0.03
    winsize : int, optional
        behaviour will only be considered walking if  >=75% of the samples within a window centered
        around the current frame are also above the threshold., by default 100 (at 400 Hz)
        choose 4 if data is already downsampled to 16 Hz

    Returns
    -------
    pandas DataFrame
        same DataFrame with additional field "walk"
    """
    walk = flow_data["velForw"] >= thres_walk
    walk = np.logical_and(np.convolve(walk, np.ones(winsize)/winsize, mode="same") >= 0.75, walk)
    flow_data["walk"] = walk
    return flow_data

def resting(flow_data, thres_rest=0.01, winsize=400):
    """apply a threshold to check whether the fly is resting or not.
    adds field "rest" to the dataframe.

    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing velocities computed from optical flow, e.g. output of compute_velocities()
        Ideally already filtered using filter_velocities()
    thres_rest : float, optional
        threshold to apply on forward walking units (rotations/s), by default 0.03
    winsize : int, optional
        behaviour will only be considered resting if  >=75% of the samples within a window centered
        around the current frame are also above the threshold., by default 400 (at 400 Hz)
        choose 16 if data is already downsampled to 16 Hz

    Returns
    -------
    pandas DataFrame
        same DataFrame with additional field "rest"
    """
    rest = np.logical_and.reduce((np.abs(flow_data["velForw"]) <= thres_rest, 
                                  np.abs(flow_data["velSide"]) <= thres_rest, 
                                  np.abs(flow_data["velTurn"]) <= thres_rest))
    rest = np.logical_and(np.convolve(rest, np.ones(winsize)/winsize, mode="same") >= 0.75, rest)
    flow_data["rest"] = rest
    return flow_data

def clean_rest(rest, N_clean=16*5):  # for 16 Hz. for full res data 400*5
    """remove short resting periods and the beginning of resting periods.
    This is usefull when considering resting in parallel to the GCaMP decay.

    Parameters
    ----------
    rest : numpy array
        binary time signal indicating whether a fly is resting or not
    N_clean : int, optional
        how many samples at the beginning of each resting period to cut off, by default 16*5 (at 16 Hz)

    Returns
    -------
    numpy array
        binary array with cleaned resting signal
    """
    rest_cleaned = np.zeros_like(rest)
    i_start = np.where(np.diff(rest.astype(int))==1)[0]
    if rest[0]:
        i_start = np.concatenate(([-1], i_start))
    i_end = np.where(np.diff(rest.astype(int))==-1)[0]
    if rest[-1]:
        i_end = np.concatenate((i_end, [len(rest)-1]))
    for this_start, this_end in zip(i_start, i_end):
        if this_start + N_clean < this_end:
            rest_cleaned[this_start+N_clean+1:this_end+1] = True
    return rest_cleaned

def fractions_walking_resting(flow_data):
    """compute fractions of walking and resting based on the flow_data DataFrame

    Parameters
    ----------
    flow_data : pandas DataFrame
        dataframe containing "rest" and "walk" fields

    Returns
    -------
    float
        fraction of walking
    float
        fraction of resting
    """
    fraction_walk = np.mean(flow_data.walk)
    fraction_rest = np.mean(flow_data.rest)
    return fraction_walk, fraction_rest

def get_opflow_df(beh_trial_dir, index_df=None, df_out_dir=None, block_error=False,
                  winsize=80, thres_walk=0.03, thres_rest=0.01, return_walk_rest=False):
    """read the optical flow data from file, load it into a dataframe, pre-process it.
    If index_df is given, load it into the pre-specified data frame

    Parameters
    ----------
    beh_trial_dir : str
        absolute path of the folder where the optic flow file "OptFlow.txt" is located
    index_df : pandas Dataframe or str, optional
        pandas dataframe or path of pickle containing dataframe to which the optic flow result is added.
        This could, for example, be a dataframe that contains indices for synchronisation with 2p data,
        by default None
    df_out_dir : str, optional
        where to save the dataframe, by default None
    block_error : bool, optional
        if True, ignore if optical flow file is not found and return an empty dataframe, by default False
    winsize : int, optional
        window size for moving average filter to be applied to raw sensor values, by default 80
    thres_walk : float, optional
        threshold to classify forward walking. unit: rotations/s, by default 0.03
    thres_rest : float, optional
        threshold to classify resting. unit: rotations/s, by default 0.01
    return_walk_rest : bool, optional
        if True, return dataframe, fraction_walking, fraction_resting.
        Otherwise, just the dataframe, by default False

    Returns
    -------
    pandas DataFrame
        optical flow dataframe with following (additional) fields:
        sens0X, sens0Y, sens1X, sens1Y, velForw, velSide, velTurn, rest, walk

    Raises
    ------
    FileNotFoundError
        if optical flow file not found and block_error == False
    SyntaxError
        Error during reading of the optical flow file and/or replacing it with empty array.
    ValueError
        length of index_df and optical flow file are not corresponding to each other by more than 10 samples
    """
    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    elif index_df is not None:
        assert isinstance (index_df, pd.DataFrame)
    try:
        flow_data = load_opflow(beh_trial_dir)
        flow_data = calibrate_opflow(flow_data)
        flow_data = compute_velocities(flow_data)
    except FileNotFoundError:
        if not block_error:
            raise FileNotFoundError("No optical flow file found in "+beh_trial_dir)
        elif index_df is not None:
            print("No optical flow file found in "+beh_trial_dir)
            empty_data = {
                "sens0X": [np.nan for _ in range(len(index_df))],
                "sens0Y": [np.nan for _ in range(len(index_df))],
                "sens1X": [np.nan for _ in range(len(index_df))],
                "sens1Y": [np.nan for _ in range(len(index_df))],
                "OpflowTime": [np.nan for _ in range(len(index_df))],
                "velForw": [np.nan for _ in range(len(index_df))],
                "velSide": [np.nan for _ in range(len(index_df))],
                "velTurn": [np.nan for _ in range(len(index_df))]
            }
            flow_data = pd.DataFrame(empty_data)  # , index=index_df.index)
        else:
            print("No optical flow file found in "+beh_trial_dir)
            empty_data = {
                "sens0X": [],
                "sens0Y": [],
                "sens1X": [],
                "sens1Y": [],
                "OpflowTime": [],
                "velForw": [],
                "velSide": [],
                "velTurn": []
            }
            flow_data = pd.DataFrame(empty_data)
    except:
        raise SyntaxError("A problem occured while loading the optical flow data of "+beh_trial_dir)

    if index_df is not None:
        if len(index_df) != len(flow_data):
            if np.abs(len(index_df) - len(flow_data)) <=10:
                Warning("Number of Thorsync ticks and length of text file do not match. \n"+\
                        "Thorsync has {} ticks, txt file has {} lines. \n".format(len(index_df), len(flow_data))+\
                        "Trial: "+beh_trial_dir)
                print("Difference: {}".format(len(index_df) - len(flow_data)))
                length = np.minimum(len(index_df), len(flow_data))
                index_df = index_df.iloc[:length, :]
                flow_data = flow_data.iloc[:length, :]
            else:
                raise ValueError("Number of Thorsync ticks and length of text file do not match. \n"+\
                        "Thorsync has {} ticks, txt file has {} lines. \n".format(len(index_df), len(flow_data))+\
                        "Trial: "+beh_trial_dir)
        df = index_df
        for key in list(flow_data.keys()):
            df[key] = flow_data[key].values
    else:
        df = flow_data
    if winsize is not None:
        df = filter_velocities(df, winsize=winsize)
    if thres_walk is not None:
        df = forward_walking(df, thres_walk)
    if thres_rest is not None:
        df = resting(df, thres_rest)
    if df_out_dir is not None:
        df.to_pickle(df_out_dir)
    if return_walk_rest:
        return df, fractions_walking_resting(df)
    return df

def get_opflow_in_twop_df(opflow_df, twop_df, twop_df_out_dir=None, thres_walk=0.03, thres_rest=0.01):
    """add downsampled optic flow information to the two photon dataframe

    Parameters
    ----------
    opflow_df : pandas DataFrame or str
        optical flow dataframe, as obtained from get_opflow_df()
    twop_df : pandas DataFrame or str
        two photon dataframe as obtained from synchronisation.get_synchronised_trial_dataframes()
    twop_df_out_dir : str, optional
        where to save the twop_df to after adding optic flow variables, by default None
    thres_walk : float, optional
        threshold to classify forward walking. unit: rotations/s, by default 0.03
    thres_rest : float, optional
        threshold to classify resting. unit: rotations/s, by default 0.01

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(opflow_df, str) and os.path.isfile(opflow_df):
        opflow_df = pd.read_pickle(opflow_df)
    assert isinstance (opflow_df, pd.DataFrame)
    if isinstance(twop_df, str) and os.path.isfile(twop_df):
        twop_df = pd.read_pickle(twop_df)
    assert isinstance (twop_df, pd.DataFrame)

    twop_index = opflow_df["twop_index"]
    N_frames = np.max(twop_index)
    twop_df = twop_df[:N_frames]
    twop_df["velForw"] = reduce_during_2p_frame(twop_index, opflow_df["velForw"])[:N_frames]
    twop_df["velSide"] = reduce_during_2p_frame(twop_index, opflow_df["velSide"])[:N_frames]
    twop_df["velTurn"] = reduce_during_2p_frame(twop_index, opflow_df["velTurn"])[:N_frames]
    twop_df["walk_resamp"] = reduce_during_2p_frame(twop_index, opflow_df["walk"], function=reduce_min)[:N_frames]
    twop_df["rest_resamp"] = reduce_during_2p_frame(twop_index, opflow_df["rest"], function=reduce_min)[:N_frames]

    fs = np.mean(1 / np.diff(twop_df["t"]))

    twop_df = resting(twop_df, thres_rest=thres_rest, winsize=int(np.floor(fs)))
    twop_df = forward_walking(twop_df, thres_walk=thres_walk, winsize=int(np.floor(fs/4)))

    if twop_df_out_dir is not None:
        twop_df.to_pickle(twop_df_out_dir)
    return twop_df
