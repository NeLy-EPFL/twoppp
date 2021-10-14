import os, sys

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp import load

date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)
all_trial_dirs = load.get_trials_from_fly(fly_dir=fly_dirs)

fly_dir = fly_dirs[0]
trial_dirs = all_trial_dirs[0]

params = PreProcessParams()
preprocess = PreProcessFly(fly_dir, params=params)

processed_dir = preprocess.trial_processed_dirs[0]

preprocess._denoise_trial_trainNinfer(processed_dir=processed_dir)

pass