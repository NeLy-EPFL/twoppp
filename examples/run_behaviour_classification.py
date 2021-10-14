"""
Classify behaviours using a classifier built by FLorian.
Use with caution. Might not generalise well to unseen flies.
DeepFly3D and post processing have to be run before.
"""
import os

from twoppp import load
from twoppp.behaviour.classification import compute_features, classify_behaviour
from twoppp.behaviour.classification import split_result_to_trials

if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
    fly_dir = load.get_flies_from_datedir(date_dir)[0]
    features_out_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER,"wavelet_behaviour_features.pkl")
    labels_out_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER, "behaviour_labels.pkl")

    compute_features(fly_dir, features_out_dir)

    classify_behaviour(features_out_dir, labels_out_dir)

    trial_dirs = load.get_trials_from_fly(fly_dir)[0]
    label_dirs = [os.path.join(trial_dir, "behData", "behaviour_labels.pkl")
                  for trial_dir in trial_dirs]
    split_result_to_trials(labels_out_dir, label_dirs, to_csv=True)
