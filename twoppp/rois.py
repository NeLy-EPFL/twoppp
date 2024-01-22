"""
sub module with utilties for ROI extraction
"""
import os
import pickle
from typing import List
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, disk, binary_erosion
from skimage.filters import threshold_local, threshold_otsu
from numpy.random import default_rng
rng = default_rng(seed=1234567890)

from twoppp.utils import get_stack, save_stack, readlines_tolist, list_join
from twoppp.analysis import pca
from twoppp import load

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
        roi_center = [int(line[-8:-5]), int(line[-3:])]
        if not any([roi_centers[i_roi] == roi_center for i_roi in range(len(roi_centers))]):
            roi_centers.append(roi_center)
        else:
            print(f"ROI center duplicate in {filename}: {roi_center}")
    N_rois = len(roi_centers)
    print("Read the centers of {} ROIs from file".format(N_rois))
    return roi_centers

def get_roi_signals_df(stack, roi_center_filename, size=(7,11), pattern="default",
                       index_df=None, df_out_dir=None, mask_out_dir=None, raw=False):
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

    raw : bool, optional
        whether the data handed over is raw, i.e. not denoised with DeepInterpolation

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
            assert isinstance(index_df, pd.DataFrame)
        if N_samples > len(index_df) and raw:
            roi_signals = roi_signals[30:-30]
            N_samples -= 60
        if len(index_df) -60 == N_samples:
            N_samples += 60
            roi_signals_new = np.zeros((N_samples, N_rois))
            roi_signals_new[:] = np.nan
            roi_signals_new[30:-30,:] = roi_signals
            roi_signals = roi_signals_new
            print("Warning: ROI signals were shorter than the DataFrame (likely because of denoising): setting first 30 and last 30 samples in df to 'np.nan'.")
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
        if raw:
            df["neuron_{}".format(i_roi)] = roi_signals[:, i_roi]
        else:
            df["neuron_denoised_{}".format(i_roi)] = roi_signals[:, i_roi]
    if df_out_dir is not None:
        df.to_pickle(df_out_dir)
    return df

def make_roi_map(roi_mask, values, setnan=False):
    roi_map = np.zeros_like(roi_mask)
    N_rois = len(np.unique(roi_mask)) - 1
    assert N_rois == values.shape[0]
    if setnan:
        roi_map[:] = np.nan
    for i_roi in range(N_rois):
        roi_map[roi_mask == (i_roi+1)] = values[i_roi]
    return roi_map

def widen_roi_map(roi_mask, spread=5):
    return dilation(roi_mask, selem=disk(spread))

def prepare_roi_selection(fly_dir: str, trial_dirs: List[str], std: str="raw", signals: str="denoised",
                          green_com_warped: str="green_com_warped.tif",
                          green_denoised: str="green_denoised.tif",
                          summary_stats: str="compare_trials.pkl",
                          out_file_name: str="green_pixels_pca_map.pkl",
                          out_plot_file_name: str="ROI_selection_pca_maps.png",
                          N_samples: int=5000) -> None:
    """
    prepares file for manual ROI selection by performing PCA on the imaging stacks.
    Also plots the resulting PCA maps.
    instead of performing PCA on the entire stack, a few pixels are sampled based on their
    standard deviation and then their PCA components are projected back into the image space.

    Parameters
    ----------
    fly_dir : str
        base directory for data related to the fly
    trial_dirs : List[str]
        directory of each trial
    std : str, optional
        [description], by default "raw"
    signals : str, optional
        [description], by default "denoised"
    green_com_warped : str
        file name of the green warped file, by default "green_com_warped.tif"
    green_denoised : str
        file name of the green denoised file, by default "green_denoised.tif"
    summary_stats : str
        file name of the summary stats, by default "compare_trials.pkl"
    out_file_name : str
        file name of the pca maps to be calculated, by default "green_pixels_pca_map.pkl"
    out_plot_file_name : str
        file name of the plots of the pca maps to be created, by default "ROI_selection_pca_maps.png"
    N_samples : int, optional
        [description], by default 5000

    Raises
    ------
    NotImplementedError
        if signals not in ["raw", "denoised"]
    """
    fly_processed_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER)
    processed_dirs = list_join(trial_dirs, load.PROCESSED_FOLDER)
    if signals == "raw":
        greens = [get_stack(os.path.join(processed_dir, green_com_warped))
                  for processed_dir in processed_dirs]
    elif signals == "denoised":
        greens = [get_stack(os.path.join(processed_dir, green_denoised))
                  for processed_dir in processed_dirs]
    else:
        raise NotImplementedError

    with open(os.path.join(fly_processed_dir, summary_stats), "rb") as f:
        summary_dict = pickle.load(f)
    green_stds = summary_dict["green_stds"]
    green_stds_raw =  summary_dict["green_stds_raw"]

    green_std = np.mean(green_stds, axis=0)
    green_std_raw = np.mean(green_stds_raw, axis=0)

    sz_diff = (np.array(green_std_raw.shape) - np.array(green_std.shape)) // 2

    if std == "raw":
        if sz_diff[0]:
            green_stds_raw = [green_raw[sz_diff[0]:-sz_diff[0], :] for green_raw in green_stds_raw]
            green_std_raw = green_std_raw[sz_diff[0]:-sz_diff[0], :]
        if sz_diff[1]:
            green_stds_raw = [green_raw[:, sz_diff[1]:-sz_diff[1]] for green_raw in green_stds_raw]
            green_std_raw = green_std_raw[:, sz_diff[1]:-sz_diff[1]]
        std = green_std_raw
        del green_std_raw
    elif std == "denoised":
        std = green_std
    else:
        raise NotImplementedError
    if signals == "raw":
        if sz_diff[0]:
            greens = [greens[:, sz_diff[0]:-sz_diff[0], :] for green_raw in greens]
        if sz_diff[1]:
            greens = [greens[:, :, sz_diff[1]:-sz_diff[1]] for green_raw in greens]

    thres = threshold_local(np.log10(std), block_size=51, method="gaussian")
    std_log_smooth = gaussian_filter(np.log10(std), sigma=10)
    thres_global = threshold_otsu(image=std_log_smooth)  # np.median(std_log_smooth)

    mask_local = np.log10(std) > thres
    mask_global = std_log_smooth > thres_global
    mask = np.logical_and(mask_local, mask_global)
    mask_erode = binary_erosion(mask, selem=disk(2))

    def get_pixels_from_mask(stack, mask):
        pixels = np.where(mask)
        pixel_values = np.array([stack[:, pixel_y, pixel_x]
                                 for pixel_y, pixel_x in zip(pixels[0], pixels[1])]).T
        return pixel_values

    green_pixels = [get_pixels_from_mask(green, mask_erode) for green in greens]

    green_pixel_means = [np.mean(stack, axis=0) for stack in green_pixels]
    green_pixel_stds = [np.std(stack, axis=0) for stack in green_pixels]

    green_pixels_z = np.concatenate([(green_p - green_pixel_mean) / green_pixel_std
                                    for green_p, green_pixel_mean, green_pixel_std
                                    in zip(green_pixels, green_pixel_means, green_pixel_stds)],
                                    axis=0)
    green_pixel_std = np.mean(green_pixel_stds, axis=0)
    N_samples = np.minimum(N_samples, len(green_pixel_stds[0]))
    i_samples = rng.choice(np.arange(len(green_pixel_stds[0])), size=N_samples, replace=False,
                           p=green_pixel_std/np.sum(green_pixel_std))
    # i_samples_rand = rng.choice(np.arange(len(green_pixel_stds[0])), size=N_samples,replace=False)

    green_pixels_z_select = green_pixels_z[:, i_samples]

    mask_sampled = np.zeros_like(mask_erode.flatten())
    for i_m, m in enumerate(np.where(mask_erode.flatten())[0]):
        if i_m in i_samples:
            mask_sampled[m] = True
    mask_sampled = mask_sampled.reshape(mask_erode.shape)

    v_pca = pca(green_pixels_z_select, zscore=False)

    def make_pca_map(mask, i_samples, values):
        pca_map = np.zeros((mask.size))
        for i_m, m in enumerate(np.where(mask.flatten())[0]):
            if i_m in i_samples:
                pca_map[m] = values[int(np.where(i_samples==i_m)[0])]
        pca_map = pca_map.reshape(mask.shape)
        return pca_map

    pca_maps = [make_pca_map(mask_erode, i_samples, v_pca[:, i]) for i in range(6)]

    save_data = {
        "green_std": std,
        "mask": mask_erode,
        "i_samples": i_samples,
        "green_pixels_z": green_pixels_z,
        "green_pixel_means": green_pixel_means,
        "green_pixel_stds": green_pixel_stds,
        "v_pca": v_pca,
        "pca_maps": pca_maps,
    }
    with open(os.path.join(fly_processed_dir, out_file_name), "wb") as f:
        pickle.dump(save_data, f, protocol=4)

    fig, axs = plt.subplots(3,3, figsize=(9.5, 6), sharex=True, sharey=True)
    axs = axs.flatten()
    axs[0].imshow(np.log10(std),
                  clim=[np.quantile(np.log10(std), 0.5), np.quantile(np.log10(std), 0.99)])
    axs[0].set_title("standard deviation across pixels")
    axs[1].imshow(mask_erode)
    axs[1].set_title("eroded mask")

    axs[2].imshow(mask_sampled)
    axs[2].set_title("5000 samples from eroded mask")

    for i_ax, ax in enumerate(axs[3:]):
        ax.imshow(gaussian_filter(pca_maps[i_ax], sigma=3),
                  cmap=plt.cm.get_cmap("seismic"), clim=[-1e-2, 1e-2])
        ax.set_title(f"PC {i_ax}")
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(fly_processed_dir, out_plot_file_name), dpi=300)
