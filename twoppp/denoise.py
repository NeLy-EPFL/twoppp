import os, sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
import pickle

from twoppp.utils import get_stack, save_stack


def get_illumination_correction(green_mean):
    """assuming an occlusion on one side (along the x axis),
    compute a linear fit to the mean across y,
    return the inverse of the fit as correction

    Parameters
    ----------
    green_mean : numpy array
        mean of green channel across time

    Returns
    -------
    correction: numpy array
        to be applied with correct_illumination()
    """
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
    """apply illumination correction in x direction
    previously computed with get_illumination_correction()

    Parameters
    ----------
    stack : numpy array
        imaging stack

    correction : numpy array
        computed using get_illumination_correction()

    Returns
    -------
    corrected: numpy array
    """
    return stack*correction

def prepare_corrected_data(train_data_tifs, out_data_tifs, fly_dir, summary_dict_pickle):
    """prepare illumination corrected data for denoising with DeepInterpolation.
    Fit straight line to decay in average brightness across the x direction
    and apply the inverse of it as correction.

    Parameters
    ----------
    train_data_tifs : list of (numpy array of str)
        stacks to be used for training

    out_data_tifs : list of str
        where to save the corrected data to

    fly_dir : str
        not used

    summary_dict_pickle : str
        path to a pickled file containing a dictionary with entry "green_means_raw"
        This will be used to fit the correction
    """
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
