import os
import sys
from datetime import datetime
from tqdm import tqdm
from glob import glob

import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

FILE_PATH = os.path.realpath(__file__)
PLOT_PATH, _ = os.path.split(FILE_PATH)
TWOPPP_PATH, _ = os.path.split(PLOT_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
sys.path.append(MODULE_PATH)
OUTPUT_PATH = os.path.join(MODULE_PATH, "outputs")

from twoppp import utils, load, rois
from twoppp import high_caff_flies, high_caff_main_fly, low_caff_main_fly, sucr_main_fly
from twoppp.plot.videos import make_multiple_video_raw_dff_beh


def make_wave_video(flies, trials, output_dir, video_name, t_range=[-15, 15]):
    all_selected_frames = []
    greens = []
    reds = []
    trial_dirs = []
    beh_dirs = []
    sync_dirs = []
    texts = []
    masks = []
    norm_greens = []
    i_fly = 0
    old_i_fly = -1
    for fly, trial in tqdm(zip(flies, trials)):
        if fly.i_fly != old_i_fly:
            i_fly += 1
            old_i_fly = fly.i_fly
        wave_details_file = fly.wave_details[:-4] + f"_{trial}.pkl"
        if not os.path.isfile(wave_details_file):
            print("Could not find file: ", wave_details_file)
            continue
        with open(wave_details_file, "rb") as f:
            wave_details = pickle.load(f)
        selected_frames = np.arange(wave_details["i_global_max"] + fly.fs*t_range[0],
                                    wave_details["i_global_max"] + fly.fs*t_range[1])
        all_selected_frames.append(selected_frames)
        greens.append(fly.trials[trial].green_raw)
        reds.append(fly.trials[trial].red_raw)
        trial_dirs.append(fly.trials[trial].dir)
        beh_dirs.append(fly.trials[trial].beh_dir)
        sync_dirs.append(fly.trials[trial].sync_dir)
        texts.append(f"fly {i_fly}")
        # str(fly.date) + " fly " + str(fly.i_flyonday) + " " + fly.trials[trial].name)
        mask = utils.get_stack(fly.mask_fine) > 0
        masks.append(mask)

        if os.path.isfile(fly.trials[trial].green_norm):
            print("Loading pre-computed normalised green data.")
            green_norm = utils.get_stack(fly.trials[trial].green_norm)
        else:
            green_denoised = utils.get_stack(fly.trials[trial].green_denoised)
            q_low = np.quantile(green_denoised, 0.005, axis=0)
            q_high = np.quantile(green_denoised, 0.995, axis=0)
            green_norm = np.clip((green_denoised-q_low) / (q_high-q_low), a_min=0, a_max=1)
            utils.save_stack(fly.trials[trial].green_norm, green_norm)
        norm_greens.append(green_norm)


    make_multiple_video_raw_dff_beh(dffs=norm_greens, trial_dirs=trial_dirs, out_dir=output_dir,
                                    video_name=video_name, beh_dirs=beh_dirs, sync_dirs=sync_dirs,
                                    camera=6, greens=greens, reds=reds, mask=masks, text=texts, text_loc="beh",
                                    select_frames=all_selected_frames, share_lim=False, time=False,
                                    vmin=0, vmax=1, colorbarlabel="")

def make_behaviour_video(fly, i_trials, output_dir, video_name, start_time=0, video_length=None, fs=16.2):
    if not isinstance(start_time, list) and not start_time:
        if video_length is None:
            selected_frames = None
    elif not isinstance(start_time, list):
        start_frame = int(start_time*fs)
        N_frames = int(video_length*fs)
        selected_frames = [np.arange(start_frame, start_frame+N_frames) for i_trial in i_trials]
    elif isinstance(start_time, list):
        if video_length is None:
            raise ValueError("please specify a video length when using a start time > 0")
        else:
            start_frames = [int(s*fs) for s in start_time]
            N_frames = int(video_length*fs)
            selected_frames = [np.arange(start_frame, start_frame+N_frames)
                               for start_frame in start_frames]
    make_multiple_video_raw_dff_beh(
        dffs=[fly.trials[i_trial].dff for i_trial in i_trials],
        trial_dirs=[fly.trials[i_trial].dir for i_trial in i_trials],
        out_dir=output_dir,
        video_name=video_name,
        beh_dirs=[fly.trials[i_trial].beh_dir for i_trial in i_trials],
        sync_dirs=[fly.trials[i_trial].sync_dir for i_trial in i_trials],
        camera=6,
        greens=[fly.trials[i_trial].green_raw for i_trial in i_trials],
        reds=[fly.trials[i_trial].red_raw for i_trial in i_trials],
        mask=utils.get_stack(fly.mask_fine)>0,
        share_mask=True,
        text=["before feeding", "during feeding", "right after feeding", "25 min after feeding"],
        text_loc="beh",
        share_lim=not "high" in fly.condition,
        vmax=600 if "sucr" in fly.condition else None,
        time=False,
        downsample=2,
        colorbarlabel="dff",
        select_frames=selected_frames
        )

def make_longterm_functional_video():
    txt_file_dir = os.path.join(load.NAS2_DIR_JB, "longterm", "longterm_T1_videos")
    fly1 = {
        "name": "longterm_T1_0930_fly5",
        "trial_names": ["1 dpi", "5 dpi", "10 dpi"],  # "8 dpi",
        "trial_txt": os.path.join(txt_file_dir, "fly_0930_trial_dirs.txt"),
        "beh_trial_txt": os.path.join(txt_file_dir, "fly_0930_beh_trial_dirs.txt"),
        "start_times_s": [85, 65, 60],  # [116, 30, 5],  # TODO
        "video_length_s": 15,
        "f_s": 10.74
    }
    fly2 = {
        "name": "longterm_T1_0928_fly4",
        "trial_names": ["1 dpi", "5 dpi"],  # , "8 dpi"],
        "trial_txt": os.path.join(txt_file_dir, "fly_0928_trial_dirs.txt"),
        "beh_trial_txt": os.path.join(txt_file_dir, "fly_0928_beh_trial_dirs.txt"),
        "start_times_s": [0, 0, 0],
        "video_length_s": 30,
        "f_s": 10.74
    }
    flies = [fly1]  # , fly2]

    green_name = "green.tif"
    red_name = "red.tif"
    i_cam = 5
    FULL_VIDEO = False
    SELECTED_TRIALS = [0,1,4]
    VIDEO_TYPE = "short_final"

    for i_fly, fly in enumerate(flies):
        trial_dirs = utils.readlines_tolist(fly["trial_txt"])
        beh_trial_dirs = utils.readlines_tolist(fly["beh_trial_txt"])

        if len(SELECTED_TRIALS):
            trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in SELECTED_TRIALS]
            beh_trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(beh_trial_dirs) if i_trial in SELECTED_TRIALS]

        # initialise file structure
        trial_processed_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER)
                                for trial_dir in trial_dirs]
        _ = [utils.makedirs_safe(trial_processed_dir)
             for trial_processed_dir in trial_processed_dirs]
        green_dirs = [os.path.join(trial_processed_dir, green_name)
                      for trial_processed_dir in trial_processed_dirs]
        red_dirs = [os.path.join(trial_processed_dir, red_name)
                    for trial_processed_dir in trial_processed_dirs]

        # convert raw to tiff if necessary
        _ = [load.convert_raw_to_tiff(trial_dir=trial_dir, green_dir=green_dir, red_dir=red_dir)
             for trial_dir, green_dir, red_dir in tqdm(zip(trial_dirs, green_dirs, red_dirs))]


        selected_frames = [np.arange(int(start*fly["f_s"]),
                                    int(start*fly["f_s"] + fly["video_length_s"]*fly["f_s"]))
                        for start in fly["start_times_s"]]
        make_multiple_video_raw_dff_beh(
            dffs=[None for _ in trial_dirs],
            trial_dirs=trial_dirs,
            out_dir=txt_file_dir,
            video_name=fly["name"]+"_"+VIDEO_TYPE,
            beh_dirs=beh_trial_dirs,
            sync_dirs=trial_dirs,
            camera=i_cam,
            greens=green_dirs,
            reds=red_dirs,
            text=None,
            time=False,
            select_frames=selected_frames if not FULL_VIDEO else None
        )



def main():
    make_longterm_functional_video()

    extend_highcaff_flies = high_caff_flies[0:1]*2 + high_caff_flies[1:2]*2 + high_caff_flies[2:3]
    wave_trials = [-2, -1, -2, -1, -2]
    """
    make_wave_video(extend_highcaff_flies, trials=wave_trials, output_dir=OUTPUT_PATH, video_name="_supvid_waves")
    """
    video_length = 30  # s
    start_times = [0, 41, 158, 128]
    make_behaviour_video(high_caff_main_fly, [0,2,3,-2], OUTPUT_PATH, video_name="_supvid_highcaff_short",
                         start_time=start_times, video_length=video_length)

    start_times = [30, 143, 163, 195]
    make_behaviour_video(low_caff_main_fly, [0,2,3,-1], OUTPUT_PATH, video_name="_supvid_lowcaff_short",
                         start_time=start_times, video_length=video_length)
    
    start_times = [101, 142, 141, 52]
    make_behaviour_video(sucr_main_fly, [0,2,3,-1], OUTPUT_PATH, video_name="_supvid_succrose_short",
                         start_time=start_times, video_length=video_length)


if __name__ == "__main__":
    main()