import os, sys

FILE_PATH = os.path.realpath(__file__)
LONGTERM_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
import pickle

from longterm.utils import get_stack, save_stack


def get_illumination_correction(green_mean):
    green_med_filt = medfilt(green_mean, kernel_size=(71,91))
    green_filt = gaussian_filter(green_med_filt, sigma=3)

    # select the area +-100 pixels from the center
    y_mean = np.mean(green_filt, axis=0)
    norm_range = [len(y_mean) // 2 - 100, len(y_mean) // 2 + 100]

    # perform linear regression in that range
    y_target = y_mean[norm_range[0]:norm_range[1]]
    x_target = np.arange(norm_range[0], norm_range[1])
    # model: y = b[0]x + b[1]
    X = np.hstack((np.expand_dims(x_target, axis=1), np.ones((len(x_target),1))))
    b = np.linalg.pinv(X.T).T.dot(y_target)
    print(f"Found correction parameters: offset={b[1]}, slope={b[0]}")
    correction = 1/(1 +  b[0]/b[1]*np.arange(len(y_mean)))
    return correction

def correct_illumination(stack, correction):
    return stack*correction

def prepare_corrected_data(train_data_tifs, out_data_tifs, fly_dir, summary_dict_pickle):
    if not isinstance(train_data_tifs, list):
        train_data_tifs = [train_data_tifs]
    if not isinstance(out_data_tifs, list):
        out_data_tifs = [out_data_tifs]
    assert len(train_data_tifs) == len(out_data_tifs)

    stacks = [get_stack(path) for path in train_data_tifs]

    with open(summary_dict_pickle, "rb") as f:
        summary_dict = pickle.load(f)
    green_mean = np.mean(summary_dict["green_means_raw"], axis=0)
    del summary_dict
    correction = get_illumination_correction(green_mean)

    stacks_corrected = [correct_illumination(stack, correction) for stack in stacks]

    for out_path, stack in zip(out_data_tifs, stacks_corrected):
        path, _ = os.path.split(out_path)
        if not os.path.isdir(path):
            os.makedirs(path)
        save_stack(out_path, stack)
