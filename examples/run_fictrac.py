import os.path

from twoppp.load import NAS2_DIR_JB, PROCESSED_FOLDER, get_trials_from_fly
from twoppp.behaviour.fictrac import config_and_run_fictrac, get_fictrac_df
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes

if __name__ == "__main__":
    fly_dirs = [
        os.path.join(NAS2_DIR_JB, "210930_PR_olfac", "Fly1"),
        os.path.join(NAS2_DIR_JB, "210930_PR_olfac", "Fly2"),
    ]
    # run fictrac for all trials
    for fly_dir in fly_dirs:
        config_and_run_fictrac(fly_dir)

    # collect output in synchronised data frames
    all_trial_dirs = get_trials_from_fly(fly_dirs)

    for i_fly, (fly_dir, trial_dirs) in enumerate(zip(fly_dirs, all_trial_dirs)):
        # get the frame times and synchronise the behavioural and two-photon recordings
        for i_trial, trial_dir in enumerate(trial_dirs):
            _, trial_name = os.path.split(trial_dir)
            trial_info = {"Date": 210930,
                          "Genotype": "PR",
                          "Fly": i_fly+1,
                          "TrialName": trial_name,
                          "i_trial": i_trial}
            beh_out_df = os.path.join(trial_dir, PROCESSED_FOLDER, "beh_df.pkl")
            twop_df, df3d_df, opflow_df = get_synchronised_trial_dataframes(
                trial_dir,
                crop_2p_start_end=30,  # select this if you used DeepInterpolation, otherwise 0
                beh_trial_dir=trial_dirs,
                sync_trial_dir=trial_dirs,
                trial_info=trial_info,
                opflow=False,
                df3d=True,  # get synchronised behavioural data frame
                opflow_out_dir=None,
                df3d_out_dir=beh_out_df,
                twop_out_dir=None)

            get_fictrac_df(trial_dir,
                index_df=beh_out_df,
                df_out_dir=beh_out_df,
                med_filt_size=5,
                sigma_gauss_size=10)
