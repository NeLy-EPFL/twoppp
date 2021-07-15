import os, sys
import numpy as np
import pandas as pd

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import find_file

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
    """
    opflow_file = find_file(beh_trial_dir, 'OptFlow.txt')
    cols = ['sens0X','sens0Y','sens1X','sens1Y','OpflowTime']
    flow_data = pd.read_table(opflow_file, sep=',', header=None, names=cols)
    return flow_data

def calibrate_opflow(flow_data, gains=gains):
    """
    This function returns the sensor values times the calibration gain.
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
    flow_data["velForw"] = opflow_filter(flow_data["velForw"], winsize)
    flow_data["velSide"] = opflow_filter(flow_data["velSide"], winsize)
    flow_data["velTurn"] = opflow_filter(flow_data["velTurn"], winsize)
    return flow_data

def forward_walking(flow_data, thres_walk=0.03, winsize=100):  # 4 if downsampled to 16 Hz
    walk = flow_data["velForw"] >= thres_walk
    walk = np.logical_and(np.convolve(walk, np.ones(winsize)/winsize, mode="same") >= 0.75, walk)
    flow_data["walk"] = walk
    return flow_data

def resting(flow_data, thres_rest=0.01, winsize=400):  # 16 if downsampled to 16 Hz
    rest = np.logical_and.reduce((np.abs(flow_data["velForw"]) <= thres_rest, 
                                  np.abs(flow_data["velSide"]) <= thres_rest, 
                                  np.abs(flow_data["velTurn"]) <= thres_rest))
    rest = np.logical_and(np.convolve(rest, np.ones(winsize)/winsize, mode="same") >= 0.75, rest)
    flow_data["rest"] = rest
    return flow_data

def fractions_walking_resting(flow_data):
    fraction_walk = np.mean(flow_data.walk)
    fraction_rest = np.mean(flow_data.rest)
    return fraction_walk, fraction_rest

def get_opflow_df(beh_trial_dir, index_df=None, df_out_dir=None, block_error=False, 
                  winsize=80, thres_walk=0.03, thres_rest=0.01, return_walk_rest=False):
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
                        "Thosync has {} ticks, txt file has {} lines. \n".format(len(index_df), len(flow_data))+\
                        "Trial: "+beh_trial_dir)
                print("Difference: {}".format(len(index_df) - len(flow_data)))
                length = np.minimum(len(index_df), len(flow_data))
                index_df = index_df.iloc[:length, :]
                flow_data = flow_data.iloc[:length, :]
            else:
                raise ValueError("Number of Thorsync ticks and length of text file do not match. \n"+\
                        "Thosync has {} ticks, txt file has {} lines. \n".format(len(index_df), len(flow_data))+\
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
