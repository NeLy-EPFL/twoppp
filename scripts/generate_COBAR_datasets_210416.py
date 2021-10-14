import os, sys
import numpy as np
import pandas as pd
import glob
import pickle

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load
from longterm.behaviour.synchronisation import get_frame_times_indices


def generate_neural_dataframe(fly_dir, out_dir):
    trial_dirs = load.get_trials_from_fly(fly_dir)[0]
    date_dir, fly = os.path.split(fly_dir)
    _, date_genotype = os.path.split(date_dir)
    date = int(date_genotype[:6])
    genotype = date_genotype[7:]
    fly = int(fly[3:])

    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
    results = pd.DataFrame(index=multi_index)

    roi_signals_file = os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_signals.pkl")
    with open(roi_signals_file, "rb") as f:
        all_ROI_signals = pickle.load(f)


    for i_trial, trial_dir in enumerate(trial_dirs):
        print(trial_dir)

        ROI_signals = all_ROI_signals[i_trial].T

        N_samples, N_neurons = ROI_signals.shape
        column_names = ["neuron_{}".format(i) for i in range(N_neurons)]

        frames = np.arange(N_samples)
        indices = pd.MultiIndex.from_arrays(([date, ] * N_samples,  # e.g 210301
                                             [genotype, ] * N_samples,  # e.g. 'J1xCI9'
                                             [fly, ] * N_samples,  # e.g. 1
                                             [i_trial, ] * N_samples,  # e.g. 1
                                             frames
                                           ),
                                           names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
        
        frame_times_2p, frame_times_beh, beh_frame_idx = get_frame_times_indices(trial_dir, crop_2p_start_end=30)

        trial_results = pd.DataFrame(index=indices)
        trial_results["t"] = frame_times_2p
        trial_results[column_names] = ROI_signals
        

        results = results.append(trial_results)

    results.to_pickle(out_dir)



def generate_behaviour_dataframe(fly_dir, out_dir):
    trial_dirs = load.get_trials_from_fly(fly_dir)[0]
    date_dir, fly = os.path.split(fly_dir)
    _, date_genotype = os.path.split(date_dir)
    date = int(date_genotype[:6])
    genotype = date_genotype[7:]
    fly = int(fly[3:])

    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
    results = pd.DataFrame(index=multi_index)

    for i_trial, trial_dir in enumerate(trial_dirs):
        print(trial_dir)
        angles_file = glob.glob(os.path.join(trial_dir, "behData", "images", "df3d", "joint_angles*"))[0]
        with open(angles_file, "rb") as f:
            angles = pickle.load(f)
        leg_keys = []
        _ = [leg_keys.append(key) for key in angles.keys()]
        angle_keys = []
        _ = [angle_keys.append(key) for key in angles[leg_keys[0]].keys()]
        name_change = {"yaw": "Coxa_yaw", 
                       "pitch": "Coxa", 
                       "roll": "Coxa_roll", 
                       "th_fe": "Femur", 
                       "roll_tr": "Femur_roll", 
                       "th_ti": "Tibia", 
                       "th_ta": "Tarsus"}
        N_features = len(leg_keys) * len(angle_keys)
        N_samples = len(angles[leg_keys[0]][angle_keys[0]])
        X = np.zeros((N_samples, N_features), dtype="float64")
        X_names = []
        for i_leg, leg in enumerate(leg_keys):
            for i_angle, angle in enumerate(angle_keys):
                X[:, i_angle + i_leg*len(angle_keys)] = np.array(angles[leg][angle])
                X_names.append("angle_" + leg + "_" + name_change[angle])

        joints_file = glob.glob(os.path.join(trial_dir, "behData", "images", "df3d", "aligned_pose*"))[0]
        with open(joints_file, "rb") as f:
            joints = pickle.load(f)

        leg_keys = []
        _ = [leg_keys.append(key) for key in joints.keys()]
        joint_keys = []
        _ = [joint_keys.append(key) for key in joints[leg_keys[0]].keys()]
        N_features = len(leg_keys) * len(joint_keys)
        Y = np.zeros((N_samples, N_features*3), dtype="float64")
        Y_names = []
        for i_leg, leg in enumerate(leg_keys):
            for i_joint, joint in enumerate(joint_keys):
                # Y[:, i_joint*3 + i_leg*len(joint_keys)*3:, :] = np.array(joints[leg][joint]["raw_pos_aligned"])
                Y[:, i_leg*len(joint_keys)*3 + i_joint*3 : i_leg*len(joint_keys)*3 + (i_joint+1)*3] = np.array(joints[leg][joint]["raw_pos_aligned"])
                #TODO: ask about this: 'fixed_pos_aligned', 'raw_pos_aligned', 'mean_length'
                #TODO: which one is x/y/z?
                Y_names += ["joint_" + leg + "_" + joint + i_ax for i_ax in ["_x", "_y", "_z"]]

        frames = np.arange(N_samples)
        indices = pd.MultiIndex.from_arrays(([date, ] * N_samples,  # e.g 210301
                                             [genotype, ] * N_samples,  # e.g. 'J1xCI9'
                                             [fly, ] * N_samples,  # e.g. 1
                                             [i_trial, ] * N_samples,  # e.g. 1
                                             frames
                                           ),
                                           names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
        trial_results = pd.DataFrame(index=indices)

        frame_times_2p, frame_times_beh, beh_frame_idx = get_frame_times_indices(trial_dir, crop_2p_start_end=30)
        trial_results["t"] = frame_times_beh
        trial_results["twop_index"] = beh_frame_idx

        trial_results[X_names] = X
        trial_results[Y_names] = Y

        results = results.append(trial_results)
    behaviour_labels = pd.read_pickle(os.path.join(fly_dir, load.PROCESSED_FOLDER, "behaviour_labels.pkl"))
    results = pd.concat((results, behaviour_labels), axis=1)
    results.to_pickle(out_dir)

if __name__ == "__main__":
    fly_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9", "Fly1")
    out_file = os.path.join(fly_dir, load.PROCESSED_FOLDER, "COBAR_neural.pkl")
               
    # generate_neural_dataframe(fly_dir, out_file)
    out_file = os.path.join(fly_dir, load.PROCESSED_FOLDER, "COBAR_behaviour.pkl")
    generate_behaviour_dataframe(fly_dir, out_file)

    beh_df = pd.read_pickle(out_file)
    pass