"""
sub-module to analyse wheel movements based on dots visible in the side view.
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import multiprocessing
import subprocess
import signal
import glob
from scipy.ndimage import gaussian_filter1d, median_filter
from time import sleep
import matplotlib.pyplot as plt
import pickle

from twoppp import load, utils
from twoppp.behaviour.fictrac import get_mean_image

f_s = 100
r_wheel = 5

def get_wheel_parameters(video_file, skip_existing=True, output_dir=None, y_min=240):
    locations_file = os.path.join(output_dir, "wheel_locations.pkl")
    if os.path.isfile(locations_file) and not skip_existing:
        with open(locations_file, "rb") as f:
            locations = pickle.load(f)
        return locations
    print("Computing mean image and detecting wheel boundaries.")
    mean_img = get_mean_image(video_file=video_file, skip_existing=skip_existing, output_name="camera_1_mean_image.jpg")
    N_y, N_x = mean_img.shape
    img = cv2.medianBlur(mean_img, 5)[y_min:,:]  # cut off top part of video with the fly and only keep wheel
    # canny_params = dict(threshold1 = 20, threshold2 = 20)
    # edges = cv2.Canny(img, **canny_params)
    black = np.zeros_like(img)
    extended_img = np.concatenate((img,black,black,black,black),axis=0)

    circles = cv2.HoughCircles(extended_img, cv2.HOUGH_GRADIENT, 2, minDist=200, param1=20, param2=20, minRadius=500, maxRadius=1200)
    circles = np.round(circles[0, :]).astype(int)
    x, y, r_out = circles[0]  # TODO: implement way to check which of the circles is correct instead of assuming it is the first one found
    r_in = r_out - 100

    if output_dir is not None:
        save_img = cv2.cvtColor(extended_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(save_img, (x, y), r_out, (0, 0, 255), 1)
        cv2.circle(save_img, (x, y), r_in, (0, 0, 255), 1)
        cv2.rectangle(save_img, (x - 5, y - 5), (x + 5, y + 5), (255, 128, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "camera_1_wheel_fit.jpg"), save_img)

    on_wheel = np.zeros_like(img, dtype=bool)
    angles = np.zeros_like(img, dtype=float)
    for i_x in np.arange(N_x):
        for i_y in np.arange(N_y//2):
            d = np.sqrt((i_x-x)**2+(i_y-y)**2)
            if d < r_out and d > r_in:
                on_wheel[i_y, i_x] = True
                angles[i_y, i_x] = np.dot([i_y-y, i_x-x], [0, d]) / d / d / np.pi * 180  # compute angles in °
    angles_rounded = np.round(angles * 2)  # each step is 0.5°

    if output_dir is not None:
        fig, axs = plt.subplots(2,1,figsize=(9.5,6))
        axs[0].imshow(angles_rounded)
        axs[1].imshow(angles_rounded, clim=[-5,5])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "camera_1_wheel_angles.jpg"))

    locations = []
    n_per_loc = []
    for angle in np.arange(-50,50):
        locations.append(np.logical_and(angles_rounded==angle, on_wheel))
        n_per_loc.append(np.sum(locations[-1]))
    locations = np.array(locations)
    locations = locations[np.array(n_per_loc) > np.max(n_per_loc)/2]
    if output_dir is not None:
        with open(os.path.join(output_dir, "wheel_locations.pkl"), "wb") as f:
            pickle.dump(locations, f)
    return locations

def extract_line_profile(img, locations, y_min=240):
    line = np.zeros(len(locations))
    img_cut = img[y_min:]
    for i_l, location in enumerate(locations):
        line[i_l] = np.mean(img_cut[location])
    return line

def get_wheel_speed(video_file, line_locations, y_min=240, max_shift=10):
    lines = []

    print("Read video to extract wheel patterns.")
    f = cv2.VideoCapture(video_file)
    while 1:
        rval, frame = f.read()
        if rval:
            frame = frame[:, :, 0]
            lines.append(extract_line_profile(frame, line_locations, y_min=y_min))
        else:
            break
    f.release()

    print("Compute wheel velocity from wheel patterns.")
    possible_shifts = np.arange(-max_shift,max_shift+1).astype(int)
    max_shift = np.max(np.abs(possible_shifts)) + 1
    shifts = np.zeros(len(lines))
    corrs = np.zeros((len(possible_shifts)))
    for i_l, line in enumerate(tqdm(lines[:-1])):
        next_line = lines[i_l+1]
        
        for i_s, shift in enumerate(possible_shifts):
            v1 = line[max_shift:-max_shift]
            v2 = next_line[max_shift+shift:-(max_shift-shift)]
            corrs[i_s] = v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        shifts[i_l] = possible_shifts[np.argmax(corrs)]
            
    v = shifts / 2 / 180 * np.pi * r_wheel * f_s
    return v

def get_wheel_df(v=None, video_file=None, index_df=None, df_out_dir=None, sigma_gauss_size=20):
    """save the velocity of the wheel into a data frame. if not supplied or already computed, compute the wheel velocity.
    This computation is dependent on dots being drawn on the side of the wheel and supplying the correct camera.
    If index_df is supplied, fictrac results will be added to this dataframe.

    Parameters
    ----------
    v : np.ndarray
        velocity vector. If None, will be computed. by default None

    video_file : str
        path to file of side view video with dots on side of the wheel clearly visible. only used in case v is None.

    index_df : pandas Dataframe or str, optional
        pandas dataframe or path of pickle containing dataframe to which the fictrac result is added.
        This could, for example, be a dataframe that contains indices for synchronisation with 2p data,
        by default None

    df_out_dir : str, optional
        if specified, will save the dataframe as .pkl, by default None

    sigma_gauss_size : int, optional
        width of Gaussian kernel applied to velocity and orientation, by default 20

    Returns
    -------
    pandas DataFrame
        dataframe containing the output of fictrac

    Raises
    ------
    IOError
        If fictract output file cannot be located

    ValueError
        If the length of the specified index_df and the fictrac output do not match
    """
    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    if index_df is not None:
        assert isinstance (index_df, pd.DataFrame)

    if v is None:
        print("Wheel velocity was not provided. Will compute it.")
        output_dir = os.path.dirname(video_file)
        line_locations = get_wheel_parameters(video_file, skip_existing=False, output_dir=output_dir, y_min=240)
        v = get_wheel_speed(video_file, line_locations, y_min=240, max_shift=10)

    v_filt = gaussian_filter1d(v.astype(float), sigma=sigma_gauss_size)

    if index_df is not None:
        if len(index_df) != len(v):
            if np.abs(len(index_df) - len(v)) <=10:
                Warning("Number of Thorsync ticks and length of wheel processing do not match. \n"+\
                        "Thorsync has {} ticks, wheel processing file has {} lines. \n".format(len(index_df), len(v))+\
                        "video_file: "+ video_file)
                print("Difference: {}".format(len(index_df) - len(v)))
                length = np.minimum(len(index_df), len(v))
                index_df = index_df.iloc[:length, :]
            else:
                raise ValueError("Number of Thorsync ticks and length of wheel processing file do not match. \n"+\
                        "Thorsync has {} ticks, wheel processing file has {} lines. \n".format(len(index_df), len(v) + 1)+\
                        "video_file: "+ video_file)
        df = index_df
        df["v_raw"] = v
        df["v"] = v_filt
    else:
        raise NotImplementedError("Please supply an index dataframe")

    if df_out_dir is not None:
        df.to_pickle(df_out_dir)
    return df


if __name__ == "__main__":
    trial_dirs = [
        # "/mnt/nas2/JB/221115_DfdxGCaMP6s_tdTom_CsChrimsonxPR/Fly1_part2/004_xz_wheel",
        # "/mnt/nas2/JB/221115_DfdxGCaMP6s_tdTom_CsChrimsonxPR/Fly1_part2/005_xz_wheel",
        "/mnt/nas2/JB/221115_DfdxGCaMP6s_tdTom_CsChrimsonxPR/Fly1_part2/006_xz_wheel",
        # "/mnt/nas2/JB/221117_DfdxGCaMP6s_tdTom_DNP9xCsChrimson/Fly1_part2/003_xz_wheel",
    ]

    for trial_dir in trial_dirs:
        video_file = os.path.join(trial_dir, "behData", "images", "camera_1.mp4")
        beh_df_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl")
        _ = get_wheel_df(v=None, video_file=video_file, index_df=beh_df_dir, df_out_dir=beh_df_dir, sigma_gauss_size=20)
    pass
