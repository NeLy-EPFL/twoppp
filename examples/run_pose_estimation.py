import os

from twoppp.load import get_trials_from_fly, NAS2_DIR_JB
from twoppp.behaviour.df3d import prepare_for_df3d, run_df3d, postprocess_df3d_trial

if __name__ == "__main__":
    fly_dir = os.path.join(NAS2_DIR_JB, "210908_PR_olfac", "Fly1")
    trial_dirs = get_trials_from_fly(fly_dir, startswith="0")[0]
    tmp_process_dir = prepare_for_df3d(trial_dirs=trial_dirs, videos=True, scope=2)
    run_df3d(tmp_process_dir)

    for trial_dir in trial_dirs:
        postprocess_df3d_trial(trial_dir)
