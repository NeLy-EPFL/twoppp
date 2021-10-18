import os.path
from tqdm import tqdm

from twoppp import load, utils
from twoppp.behaviour.olfaction import get_sync_signals_olfaction
from twoppp.plot.videos import make_all_odour_condition_videos

date_dir = os.path.join(load.NAS2_DIR_JB, "211005_J1M5")

fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)
all_trial_dirs = load.get_trials_from_fly(fly_dirs, endswith="xz", startswith="0")

for i_fly, (fly_dir, trial_dirs) in enumerate(zip(fly_dirs, all_trial_dirs)):
    for i_trial, trial_dir in enumerate(tqdm(trial_dirs)):
        _ = get_sync_signals_olfaction(
            trial_dir,
            sync_out_file=load.PROCESSED_FOLDER+"/sync.pkl",
            paradigm_out_file=load.PROCESSED_FOLDER+"/paradigm.pkl")

        print("MAKING STIM RESPONSE VIDEO: ", fly_dir)
        video_dirs = [os.path.join(trial_dir, "behData", "images", "camera_5.mp4")
                    for trial_dir in trial_dirs]
        paradigm_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER, "paradigm.pkl")
                        for trial_dir in trial_dirs]
        out_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER)
        make_all_odour_condition_videos(video_dirs, paradigm_dirs, out_dir,
                                        video_name="stim_responses", frame_range=[-500,1500],
                                        stim_length=1000, frame_rate=None,
                                        size=(120,-1), conditions=["WaterB"])