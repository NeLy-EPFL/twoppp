import os, sys
import pickle

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes
from twoppp.behaviour.optic_flow import get_opflow_df
from twoppp.utils import readlines_tolist
from twoppp.rois import get_roi_signals_df
fly_dirs = [
    "/mnt/NAS2/LH/210512/fly3",
    "/mnt/NAS2/LH/210514/fly1",
    "/mnt/NAS2/LH/210519/fly1",
    "/mnt/NAS2/LH/210521/fly1",
    "/mnt/NAS2/LH/210524/fly1",

    "/mnt/NAS2/LH/210526/fly2",
    "/mnt/NAS2/LH/210527/fly4",
    "/mnt/NAS2/LH/210531/fly2",
    "/mnt/NAS2/LH/210602/fly2",
    "/mnt/NAS2/LH/210603/fly2",

    "/mnt/NAS2/LH/210604/fly3",
    "/mnt/NAS2/LH/210618/fly5"
    ]

fractions = [[] for _ in fly_dirs]
fractions_out_dir = os.path.join(MODULE_PATH, "outputs", "_fractions_walking_resting_.pkl")

for i_fly, fly_dir in enumerate(fly_dirs):
    if i_fly != 6:
        continue
    print("====="+fly_dir)
    tmp, fly = os.path.split(fly_dir)
    fly = int(fly[-1:])
    _, date = os.path.split(tmp)
    date = int(date)
    trial_dirs = readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
    sync_trial_dirs = readlines_tolist(os.path.join(fly_dir, "sync_trial_dirs.txt"))
    beh_trial_dirs = readlines_tolist(os.path.join(fly_dir, "beh_trial_dirs.txt"))
    fractions[i_fly] = [(None, None) for _ in trial_dirs]
    for i_trial in range(len(trial_dirs)):
        _, trial_name = os.path.split(trial_dirs[i_trial])
        print("====="+trial_name)
        trial_info = {"Date": date,
                    "Genotype": "J1xM5",
                    "Fly": fly,
                    "TrialName": trial_name,
                    "i_trial": i_trial
                    }
        processed_dir = os.path.join(trial_dirs[i_trial], load.PROCESSED_FOLDER)
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
        opflow_out_dir = os.path.join(processed_dir, "opflow_df.pkl")
        df3d_out_dir = os.path.join(processed_dir, "beh_df.pkl")
        twop_out_dir = os.path.join(processed_dir, "twop_df.pkl")
        try:
            
            twop_df, df3d_df, opflow_df = get_synchronised_trial_dataframes(trial_dirs[i_trial], 
                                                                            crop_2p_start_end=30, 
                                                                            beh_trial_dir=beh_trial_dirs[i_trial], 
                                                                            sync_trial_dir=sync_trial_dirs[i_trial], 
                                                                            trial_info=trial_info,
                                                                            opflow=True, 
                                                                            df3d=True, 
                                                                            opflow_out_dir=opflow_out_dir, 
                                                                            df3d_out_dir=df3d_out_dir, 
                                                                            twop_out_dir=twop_out_dir)
            
            opflow_df, this_fractions = get_opflow_df(beh_trial_dirs[i_trial], index_df=opflow_out_dir, df_out_dir=opflow_out_dir, 
                                                 block_error=True, return_walk_rest=True, winsize=80)
            print("walking, resting: ", this_fractions)
            fractions[i_fly][i_trial] = this_fractions
        except:
            print("Error in trial: "+trial_dirs[i_trial])
        try:
            stack = os.path.join(processed_dir, "green_denoised_t1.tif")
            # """
            roi_file_corrected = os.path.join(processed_dir, "ROI_centers_corrected.txt")
            twop_out_dir_corrected = os.path.join(processed_dir, "twop_df_corrected.pkl")
            mask_out_dir_corrected = os.path.join(processed_dir, "ROI_mask_corrected.tif")
            df = get_roi_signals_df(stack, roi_file_corrected, size=[7,11], pattern="default", index_df=twop_out_dir, 
                                    df_out_dir=twop_out_dir_corrected, mask_out_dir=mask_out_dir_corrected)
            roi_file_corrected_mean = os.path.join(fly_dir, "processed", "ROI_centers_corrected_mean.txt")
            twop_out_dir_corrected = os.path.join(processed_dir, "twop_df_corrected_mean.pkl")
            mask_out_dir_corrected = os.path.join(processed_dir, "ROI_mask_corrected_mean.tif")
            df = get_roi_signals_df(stack, roi_file_corrected, size=[7,11], pattern="default", index_df=twop_out_dir, 
                                    df_out_dir=twop_out_dir_corrected, mask_out_dir=mask_out_dir_corrected)
            # """
            roi_file = os.path.join(fly_dir, "processed", "ROI_centers.txt")
            mask_out_dir = os.path.join(fly_dir, "processed", "ROI_mask.tif")
            df = get_roi_signals_df(stack, roi_file, size=[7,11], pattern="default", index_df=twop_out_dir, 
                                    df_out_dir=twop_out_dir, mask_out_dir=mask_out_dir)

            
        except:
            continue

outdata = {"flies": fly_dirs,
           "fractions:": fractions}
with open(fractions_out_dir, "wb") as f:
    pickle.dump(outdata, f)
        