import os.path

from twoppp import rois, load
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes

if __name__ == "__main__":
    fly_dir = "path/to/fly"
    trial_dirs = load.get_trials_from_fly(fly_dir, exclude="processed")[0]
    green_denoised_stacks = [os.path.join(trial_dir, "green_denoised.tif")
                             for trial_dir in trial_dirs]
    
    roi_file = "path/to/fly/processed/ROI_centers.txt"
    mask_out_dir = "path/to/fly/processed/ROI_mask.tif"

    # this step should be replaced by manual ROI detection, but it is here to
    # illustrate how the ROI centers should be saved
    roi_centers = [
        [100, 200],  # [y_coord, x_coord]
        [300, 600]
    ]
    rois.write_roi_center_file(roi_centers, roi_file)

    for i_trial, (trial_dir, green_denoised) in enumerate(trial_dirs, green_denoised_stacks):
        _, trial_name = os.path.split(trial_dir)
        trial_info = {"Date": 211201,
                      "Genotype": "J1M5",
                      "Fly": 1,
                      "TrialName": trial_name,
                      "i_trial": i_trial}

        # get the times and synchronise the neural and two-photon recordings
        twop_out_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER, "twop_df.pkl")
        twop_df, df3d_df, opflow_df = get_synchronised_trial_dataframes(
            trial_dirs[i_trial],
            crop_2p_start_end=30,  # select this if you used DeepInterpolation, otherwise 0
            beh_trial_dir=trial_dirs,
            sync_trial_dir=trial_dirs,
            trial_info=trial_info,
            opflow=False,
            df3d=False,
            opflow_out_dir=None,
            df3d_out_dir=None,
            twop_out_dir=twop_out_dir)

        _ = rois.get_roi_signals_df(
            green_denoised,
            roi_file,
            size=(7,11),
            pattern="default",  # uses a rhomboid pattern
            index_df=twop_out_dir,
            df_out_dir=twop_out_dir,
            mask_out_dir=mask_out_dir)  # saves an image showing location and of ROIs
