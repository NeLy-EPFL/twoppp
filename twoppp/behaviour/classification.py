import os, sys
import glob

import pickle
import pandas as pd
import behavelet
from tqdm import tqdm
import sklearn.manifold
import sklearn.decomposition
from matplotlib import pyplot as plt
import numpy as np

FILE_PATH = os.path.realpath(__file__)
BEHAVIOUR_PATH, _ = os.path.split(FILE_PATH)
TWOPPP_PATH, _ = os.path.split(BEHAVIOUR_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load

def compute_features(fly_dir, features_out_dir):
    trial_dirs = load.get_trials_from_fly(fly_dir)[0]
    date_dir, fly = os.path.split(fly_dir)
    _, date_genotype = os.path.split(date_dir)
    date = int(date_genotype[:6])
    genotype = date_genotype[7:]
    fly = int(fly[3:])

    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
    results = pd.DataFrame(index=multi_index)

    for i_trial, trial_dir in enumerate(trial_dirs):
        angles_file = glob.glob(os.path.join(trial_dir, "behData", "images", "df3d", "joint_angles*"))[0]
        with open(angles_file, "rb") as f:
            angles = pickle.load(f)
        leg_keys = []
        _ = [leg_keys.append(key) for key in angles.keys()]
        angle_keys = []
        _ = [angle_keys.append(key) for key in angles[leg_keys[0]].keys()]
        N_features = len(leg_keys) * len(angle_keys)
        N_samples = len(angles[leg_keys[0]][angle_keys[0]])

        X = np.zeros((N_samples, N_features), dtype="float64")
        X_names = []
        for i_leg, leg in enumerate(leg_keys):
            for i_angle, angle in enumerate(angle_keys):
                X[:, i_angle + i_leg*len(angle_keys)] = np.array(angles[leg][angle])
                X_names.append(leg + "_" + angle)
        
        freqs, power, X_coeff = behavelet.wavelet_transform(X, n_freqs=25, fsample=100., fmin=1., fmax=50., gpu=True)
        coeff_columns = [f"Coeff {col} {freq}Hz" for col in X_names for freq in freqs]  # has to stay the same
        frames = np.arange(N_samples)
        indices = pd.MultiIndex.from_arrays(([date, ] * N_samples,  # e.g 210301
                                             [genotype, ] * N_samples,  # e.g. 'J1xCI9'
                                             [fly, ] * N_samples,  # e.g. 1
                                             [i_trial, ] * N_samples,  # e.g. 1
                                             frames
                                           ),
                                           names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
        trial_results = pd.DataFrame(index=indices)
        trial_results[coeff_columns] = X_coeff
        results = results.append(trial_results)
    
    results.to_pickle(features_out_dir)


def classify_behaviour(features_file, labels_out_dir, clf=None, le=None):
    if clf is None:
        with open(os.path.join(BEHAVIOUR_PATH, "behaviour_classifier.pkl"), "rb") as f:
            clf = pickle.load(f)
    if le is None:
        with open(os.path.join(BEHAVIOUR_PATH, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
    
    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
    results = pd.DataFrame(index=multi_index)

    wavelet_df = pd.read_pickle(features_file)
    feature_columns = [col for col in wavelet_df.columns if "Coeff" in col]
    X = wavelet_df[feature_columns].values
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)
    logproba = np.log(proba)
    entropy = np.sum(-proba * logproba, axis=1)
    prediction = le.inverse_transform(y_pred)
    d = np.stack([prediction, entropy], axis=-1)
    fly_results = pd.DataFrame(data=d, columns=["Prediction", "Entropy"], index=wavelet_df.index) 
    behaviours = le.inverse_transform(np.arange(proba.shape[1]))
    for i, behaviour in enumerate(behaviours):
        fly_results[f"Probability {behaviour}"] = proba[:, i]
    results = results.append(fly_results)

    results.to_pickle(labels_out_dir)

def split_result_to_trials(fly_labels_out_dir, trial_label_out_dirs, to_csv=False):
    all_labels = pd.read_pickle(fly_labels_out_dir)
    for i_trial, trial_label_out_dir in enumerate(trial_label_out_dirs):
        trial_labels = all_labels[all_labels.index.get_level_values("Trial") == 0]
        trial_labels.to_pickle(trial_label_out_dir)
        if to_csv:
            trial_labels.to_csv(trial_label_out_dir[:-4]+".csv")





if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
    fly_dir = load.get_flies_from_datedir(date_dir)[0]
    features_out_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER, "wavelet_behaviour_features.pkl")
    labels_out_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER, "behaviour_labels.pkl")

    # compute_features(fly_dir, features_out_dir)

    # classify_behaviour(features_out_dir, labels_out_dir)

    trial_dirs = load.get_trials_from_fly(fly_dir)[0]
    label_dirs = [os.path.join(trial_dir, "behData", "behaviour_labels.pkl")
                  for trial_dir in trial_dirs]
    split_result_to_trials(labels_out_dir, label_dirs, to_csv=True)
    pass