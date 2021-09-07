import os, sys
import numpy as np

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load

fly_dirs = [os.path.join(load.NAS_DIR_JB, "210301_J1xCI9", "Fly1"),
            os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212", "Fly1"),
            os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly2"),
            os.path.join(load.NAS_DIR, "LH", "210415", "J1M5_fly3"),
            os.path.join(load.NAS_DIR, "LH", "210423_caffeine", "J1M5_fly2"),
            os.path.join(load.NAS_DIR, "LH", "210427_caffeine", "J1M5_fly1"),
            ]
all_trial_dirs = load.get_trials_from_fly(fly_dirs)

diffs = []
for i_fly, (fly_dir, trial_dirs) in enumerate(zip(fly_dirs, all_trial_dirs)):
    diffs_fly = []
    for i_trial, trial_dir in enumerate(trial_dirs):
        processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
        com_offsets_file = os.path.join(processed_dir, "com_offset.npy")
        if os.path.isfile(com_offsets_file):
            try:
                com_offsets = np.load(com_offsets_file)
            except:
                continue
            # mins = np.min(com_offsets, axis=0)
            mins = np.quantile(com_offsets, 0.01, axis=0)
            # maxs = np.max(com_offsets, axis=0)
            maxs = np.quantile(com_offsets, 0.99, axis=0)
            diff = maxs - mins
            diffs_fly.append(diff)
            # print(diff)
            print(trial_dir)
    diffs.append(diffs_fly)
