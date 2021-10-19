import os, sys
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve

from twoppp.utils import get_stack, save_stack, readlines_tolist
from twoppp.utils.df import get_multi_index_trial_df

# copying from nely_suite because of an import error
def local_correlations(Y, eight_neighbours=True, swap_dim=False, order_mean=1):
    """
    This function was copied from CaImAn published in Giovannucci et al. eLife, 2019.
    Computes the correlation image for the input dataset Y
    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    if order_mean == 0:
        rho = np.ones(np.shape(Y)[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:, :] = rho[1:, :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:, 1:] = rho[:, 1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h ** (order_mean)
        rho[1:, :] = rho[1:, :] + rho_h ** (order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w ** (order_mean)
        rho[:, 1:] = rho[:, 1:] + rho_w ** (order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0] = neighbors[0] - 1
        neighbors[-1] = neighbors[-1] - 1
        neighbors[:, 0] = neighbors[:, 0] - 1
        neighbors[:, -1] = neighbors[:, -1] - 1
        neighbors[:, :, 0] = neighbors[:, :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:, 1:] = rho[1:, 1:] * rho_d1
                rho[1:, :-1] = rho[1:, :-1] * rho_d1
                rho[:-1, 1:] = rho[:-1, 1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2 ** (order_mean)
                rho[1:, 1:] = rho[1:, 1:] + rho_d1 ** (order_mean)
                rho[1:, :-1] = rho[1:, :-1] + rho_d1 ** (order_mean)
                rho[:-1, 1:] = rho[:-1, 1:] + rho_d2 ** (order_mean)

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 3
            neighbors[-1, :] = neighbors[-1, :] - 3
            neighbors[:, 0] = neighbors[:, 0] - 3
            neighbors[:, -1] = neighbors[:, -1] - 3
            neighbors[0, 0] = neighbors[0, 0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1, 0] = neighbors[-1, 0] + 1
            neighbors[0, -1] = neighbors[0, -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 1
            neighbors[-1, :] = neighbors[-1, :] - 1
            neighbors[:, 0] = neighbors[:, 0] - 1
            neighbors[:, -1] = neighbors[:, -1] - 1

    if order_mean == 0:
        rho = np.power(rho, 1.0 / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho

def get_roi_signals(stack, centers=None, size=(0, 0), mask=None, pattern=None, mask_out_dir=None):
    if not ((isinstance(centers, list) and isinstance(centers[0], list) and len(centers[0]) == 2) or \
           (isinstance(centers, np.ndarray) and centers.shape[1] == 2) or \
            centers is None):
        centers = [centers]
        if not len(centers[0]) == 2:
            raise ValueError("If centers is not None, it should be a list of 2D coordinates")
    stack = get_stack(stack)
    if centers is not None and pattern is None and mask is None:
        if size == [0, 0] or size == (0, 0):
            return np.array([stack[:, center[0], center[1]] for center in centers]).T
        else:
            return np.array([np.mean(stack[:, center[0]-size[0]:center[0]+size[0],
                                            center[1]-size[1]:center[1]+size[1]], axis=(1, 2)) for center in centers]).T
    elif pattern is not None:
        mask = get_roi_mask(stack,centers, size, pattern, mask_out_dir=mask_out_dir)
    if mask is not None:
        return get_roi_signals_from_mask(stack, mask)
    else:
        raise NotImplementedError

def get_roi_signals_from_mask(stack, mask):
    stack = get_stack(stack)
    n_rois = len(np.unique(mask))-1
    signals = np.zeros((stack.shape[0], n_rois))
    for i_roi, roi_num in enumerate(np.unique(mask)[1:]):
        mask_one_roi = np.where(mask == roi_num)
        for i, j in zip(mask_one_roi[0], mask_one_roi[1]):
            signals[:, i_roi] += stack[:, i, j]
        signals[:, i_roi] /= len(mask_one_roi[0])
    return signals

def get_roi_mask(stack, centers, size=(7,11), pattern="default", binary=False, mask_out_dir=None):
    assert (isinstance(centers, list) and isinstance(centers[0], list) and len(centers[0]) == 2) or \
           (isinstance(centers, np.ndarray) and centers.shape[1] == 2)
    if pattern == "default" and (size == [7, 11] or size == (7, 11)):
        pattern = np.array(
            [[0, 0, 0, 1, 0, 0, 0],# 1
            [0, 0, 1, 1, 1, 0, 0],# 2
            [0, 1, 1, 1, 1, 1, 0],# 3
            [0, 1, 1, 1, 1, 1, 0],# 4
            [1, 1, 1, 1, 1, 1, 1],# 5
            [1, 1, 1, 1, 1, 1, 1],# 6
            [1, 1, 1, 1, 1, 1, 1],# 7
            [0, 1, 1, 1, 1, 1, 0],# 8
            [0, 1, 1, 1, 1, 1, 0],# 9
            [0, 0, 1, 1, 1, 0, 0],# 10
            [0, 0, 0, 1, 0, 0, 0]]# 11
            )#1, 2, 3, 4, 5, 6, 7
    elif pattern == "default" and (size == [5, 9] or size == (5, 9)):
        pattern = np.array(
            [[0, 0, 1, 0, 0],# 1
            [0, 1, 1, 1, 0],# 2
            [0, 1, 1, 1, 0],# 3
            [1, 1, 1, 1, 1],# 4
            [1, 1, 1, 1, 1],# 5
            [1, 1, 1, 1, 1],# 6
            [0, 1, 1, 1, 0],# 7
            [0, 1, 1, 1, 0],# 8
            [0, 0, 1, 0, 0]]# 9
            )#1, 2, 3, 4, 5, 6, 7
    elif pattern == "default" and (size == [3, 5] or size == (3, 5)):
        pattern = np.array(
            [[0, 1, 0],# 1
            [1, 1, 1],# 2
            [1, 1, 1],# 3
            [1, 1, 1],# 4
            [0, 1, 0]]# 5
            )#1, 2, 3
    if not isinstance(pattern, np.ndarray):
        raise NotImplementedError

    mask = np.zeros((stack.shape[1:]))
    for i_c, center in enumerate(centers):
        # give each pixel in the ROI center a unique uneven number
        mask[center[0], center[1]] = i_c*2 + 1
    # convolve with ROI shape as kernel
    mask = convolve(mask, pattern)
    # remove all pixels where 2 ROIs overlap 
    # all ROIs have uneven numbers. if they overlap, the pixel value is even
    mask[mask % 2 == 0] = 0
    # bring pixel values back o values corresponding to ROI numbers
    mask = (mask + 1) // 2
    assert np.max(mask) <= len(centers)
    assert all([np.sum(mask==i) <= np.sum(pattern) for i in range(1,len(centers)+1)])

    if binary:
        mask = mask > 0
    if mask_out_dir is not None:
        save_stack(mask_out_dir, mask)
    return mask
            
def get_dff_from_traces(signals, length_baseline=10, return_f0=False, f0_min=None, lift_neg_baseline=None):
    """
    Extract traces for individual neurons for a given mask with non overlapping connected components.
    Parameters
    ----------
    signals : numpy array
        Image stack with fluorescence values. First dimension is time.

    length_baseline : int, optional
        Length of the baseline used to compute dF/F.
        Default is 10.

    return_f0 : bool
        whether to return the baseline values as a 2nd output value
        default: False

    f0_min : int
        whether to apply a minimum baseline
        default: None

    Returns
    -------
    dff_traces : numpy array
        First dimension encodes neuron number. Second dimension is time.

    f_0s : numpy array
        optional additional output with the dff baselines
    """
    dff = np.zeros_like(signals)
    f_0s = np.zeros((signals.shape[-1]))
    for roi_num in range(signals.shape[-1]):
        signal = signals[:, roi_num]
        convolved = np.convolve(
            signal, np.ones(length_baseline), mode="valid"
        )
        f_0 = np.min(convolved) / length_baseline
        if lift_neg_baseline is not None and f_0 < lift_neg_baseline:
            f_0s[roi_num] = f_0
            signal += (lift_neg_baseline - f_0)
            f_0 = lift_neg_baseline
        else:
            f_0 = np.maximum(f_0, f0_min) if f0_min is not None else f_0
            f_0s[roi_num] = f_0
        dff[:, roi_num] = (signal - f_0) / f_0
    return dff if not return_f0 else (dff, f_0s)

def write_roi_center_file(centers, filename):
    with open(filename, "w") as f:
        for i_c, coord in enumerate(centers):
            string = "{:3}: {:3}, {:3}\n".format(i_c, coord[0], coord[1])
            f.write(string)

def read_roi_center_file(filename):
    """
    caution: this only works for files that were written with write_roi_center_file()
             and where the coordinates are 3 digits maximum
    """
    roi_centers_text = readlines_tolist(filename)
    roi_centers = []
    for line in roi_centers_text:
        roi_centers.append([int(line[-8:-5]), int(line[-3:])])
    N_rois = len(roi_centers)
    print("Read the centers of {} ROIs from file".format(N_rois))
    return roi_centers

def get_roi_signals_df(stack, roi_center_filename, size=(7,11), pattern="default",
                       index_df=None, df_out_dir=None, mask_out_dir=None):
    """extract the temporal signals from manually selected regions of interest
    using a fixed shape

    Parameters
    ----------
    stack : numpy array or str
        fluorescence data stack of images from which to extract the ROI data

    roi_center_filename : str
        .txt file written with write_roi_center_file() containing the centers of the ROIs

    size : tuple, optional
        fixed size of the ROI. Implemented sizes: (7, 11), (5, 9), (3, 5).
        by default (7,11)

    pattern : str or numpy array, optional
        currently implemented: "default" or a binary numpy array detailling the shape,
        by default "default"

    index_df : str or pandas.DataFrame, optional
        dataframe containing synchronisation data, e.g. frame time stamps, by default None

    df_out_dir : str, optional
        where to store the dataframe, by default None

    mask_out_dir : str, optional
        where to store the mask including all the ROIs, by default None

    Returns
    -------
    pandas DataFrame
        dataframe including "neuron_0" to "neuron_N" fields with ROI signals

    Raises
    ------
    ValueError
        if the supplied dataframe is much larger/shorter than the ROI signals
    """
    roi_centers = read_roi_center_file(roi_center_filename)
    roi_signals = get_roi_signals(stack, centers=roi_centers, size=size, pattern=pattern, mask=None, mask_out_dir=mask_out_dir)
    N_samples, N_rois = roi_signals.shape
    if index_df is not None:
        if isinstance(index_df, str) and os.path.isfile(index_df):
            index_df = pd.read_pickle(index_df)
        else:
            assert isinstance (index_df, pd.DataFrame)
        if len(index_df) > N_samples:
            if len(index_df) - N_samples <= 5:
                print("Difference between thorsync ticks and two photon data: {} frames \n".format(len(index_df) - N_samples)+\
                      "This might be because the denoising algorithm cuts the data to multiples of the batch size.")
                index_df = index_df.iloc[:N_samples, :]
            else:
                raise ValueError("Difference between thorsync ticks and two photon data larger than 5 frames")
        assert len(index_df) == N_samples
        df = index_df
    else:
        df = pd.DataFrame()
    for i_roi in range(N_rois):
            df["neuron_{}".format(i_roi)] = roi_signals[:, i_roi]
    if df_out_dir is not None:
        df.to_pickle(df_out_dir)
    return df
