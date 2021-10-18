import os.path

from twoppp import load
from twoppp.register.warping import warp, save_ref_frame

if __name__ == "__main__":

    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    all_trial_dirs = load.get_trials_from_fly(fly_dirs)

    for trial_dirs, fly_dir in zip(all_trial_dirs, fly_dirs):
        ref_stack = os.path.join(trial_dirs[0], load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
        ref_frame_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER, "ref_frame_com.tif")

        save_ref_frame(ref_stack, ref_frame_dir=ref_frame_dir, com_pre_reg=True)

        for trial_dir in trial_dirs:
            print("processing trial: " + trial_dir)
            processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)

            red_dir = os.path.join(processed_dir, load.RAW_RED_TIFF)
            green_dir = os.path.join(processed_dir, load.RAW_GREEN_TIFF)

            _ = load.convert_raw_to_tiff(trial_dir, overwrite=False, return_stacks=False,
                                         green_dir=green_dir, red_dir=red_dir)

            red_out_dir = os.path.join(processed_dir, "red_com_warped.tif")
            green_out_dir = os.path.join(processed_dir, "green_com_warped.tif")
            offset_dir = os.path.join(processed_dir, "com_offset.npy")

            warp(stack1=red_dir, stack2=green_dir, ref_frame=ref_frame_dir,
                 stack1_out_dir=red_out_dir, stack2_out_dir=green_out_dir,
                 com_pre_reg=True, offset_dir=offset_dir, return_stacks=False,
                 overwrite=False, select_frames=None, parallel=True, verbose=True,
                 w_output=None, initial_w=None, save_motion_field=False, param=None)
