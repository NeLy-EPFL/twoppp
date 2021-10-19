import os.path

from twoppp.load import NAS2_DIR_JB, get_trials_from_fly, PROCESSED_FOLDER
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes
from twoppp.behaviour.optic_flow import get_opflow_df, get_opflow_in_twop_df

if __name__ == "__main__":
    fly_dir = os.path.join(NAS2_DIR_JB, "211005_J1M5", "Fly1")
    trial_dirs = get_trials_from_fly(fly_dir, startswith="xz", exclude="processed")[0]

    for i_trial, trial_dir in enumerate(trial_dirs):
        _, trial_name = os.path.split(trial_dir)
        trial_info = {"Date": 211005,
                      "Genotype": "J1M5",
                      "Fly": 1,
                      "TrialName": trial_name,
                      "i_trial": i_trial}

        opflow_out_dir = os.path.join(trial_dir, PROCESSED_FOLDER, "opflow_df.pkl")
        twop_out_dir = os.path.join(trial_dir, PROCESSED_FOLDER, "twop_df.pkl")
        twop_df, df3d_df, opflow_df = get_synchronised_trial_dataframes(
            trial_dirs[i_trial],
            crop_2p_start_end=30,
            beh_trial_dir=trial_dirs,
            sync_trial_dir=trial_dirs,
            trial_info=trial_info,
            opflow=True,
            df3d=False,
            opflow_out_dir=opflow_out_dir,
            df3d_out_dir=False,
            twop_out_dir=twop_out_dir)

        opflow_df, this_fractions = get_opflow_df(trial_dir,
            index_df=opflow_out_dir,
            df_out_dir=opflow_out_dir,
            return_walk_rest=True,
            winsize=80)
        _ = get_opflow_in_twop_df(
            twop_df=twop_out_dir,
            opflow_df=opflow_df,
            twop_df_out_dir=twop_out_dir,
            thres_walk=0.03,
            thres_rest=0.01)
        print("walking, resting: ", this_fractions)
