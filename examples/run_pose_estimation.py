import os

from twoppp.load import get_trials_from_fly, NAS2_DIR_JB, PROCESSED_FOLDER
from twoppp.behaviour.df3d import prepare_for_df3d, run_df3d, postprocess_df3d_trial, get_df3d_dataframe
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes

if __name__ == "__main__":
    fly_dir = os.path.join(NAS2_DIR_JB, "210908_PR_olfac", "Fly1")
    trial_dirs = get_trials_from_fly(fly_dir, startswith="0")[0]
    tmp_process_dir = prepare_for_df3d(trial_dirs=trial_dirs, videos=True, scope=2)
    run_df3d(tmp_process_dir)

    for i_trial, trial_dir in enumerate(trial_dirs):
        # run the df3dPostProcessing for each trial
        postprocess_df3d_trial(trial_dir)

        # synchronise results to two photon recording and save them in a data frame
        _, trial_name = os.path.split(trial_dir)
        trial_info = {"Date": 210930,
                        "Genotype": "PR",
                        "Fly": 1,
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

        get_df3d_dataframe(trial_dir, index_df=beh_out_df, out_dir=beh_out_df)
