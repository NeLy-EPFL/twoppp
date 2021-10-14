# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import sys, os.path
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal import medfilt, convolve
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing
from scipy.ndimage.filters import median_filter
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import utils2p

FILE_PATH = os.path.realpath(__file__)
LONGTERM_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import get_stack, crop_img
from longterm import load
from longterm.plot.videos import make_video_dff, make_multiple_video_dff, make_multiple_video_2p

def _compute_dff(stack, baseline, apply_filter=True):
    """
    This function calculates the change in fluorescence change in percent.
    First dimension of stack is assumed to be time.
    Parameters
    ----------
    img : np.array
        Single image or stack of images.
    baseline : np.array, optional
        Must have the same dimension as the image(s) given in img.
    apply_filter : boolean
        If true the dff stacked is median filter with (3, 3, 3) kernel before
        it is returned.
    Returns
    -------
    dff_img : np.array
        dff Image.
    """
    dff_img = (
        np.divide(
            (stack - baseline),
            baseline,
            out=np.zeros_like(stack, dtype=np.double),
            where=(baseline != 0),
        )
        * 100
    )
    if apply_filter:
        dff_img = median_filter(dff_img, (3, 3, 3))
    return dff_img

def _find_pixel_wise_baseline(stack, n=10, occlusions=None):
    """
    This functions finds the indices of n consecutive frames that can serve
    as a fluorescence baseline. It convolves the fluorescence trace of each
    pixel with a rectangular signal of length n and finds the
    minimum of the convolved signal.
    Parameters
    ----------
    stack : np.array 3D
        First dimension should encode time.
        Second and third dimension are for space.
    n : int, default = 10
        Length of baseline.
    occlusions : numpy array of type boolean
        Occlusions are ignored in baseline calculation.
        Default is None.
    Returns
    -------
    baseline_img : np.array 3D
        Baseline image.
    """
    convolved = convolve1d(stack, np.ones(n), axis=0)
    if occlusions is not None:
        occ_convolved = convolve1d(occlusions, np.ones(n), axis=0)
        occluded_pixels = np.where(occ_convolved)
        convolved[occluded_pixels] = np.iinfo(convolved.dtype).max
    length_of_valid_convolution = max(stack.shape[0], n) - min(stack.shape[0], n) + 1
    start_of_valid_convolution = math.floor(n / 2)
    convolved = convolved[
        start_of_valid_convolution : start_of_valid_convolution
        + length_of_valid_convolution
    ]
    indices = np.argmin(convolved, axis=0)
    baseline_img = np.zeros(stack.shape[1:])
    for i in range(stack.shape[1]):
        for j in range(stack.shape[2]):
            baseline_i_j = np.arange(indices[i, j], indices[i, j] + n)
            baseline_img[i, j] = np.sum(stack[baseline_i_j, i, j]) / n
    return baseline_img

def _quantile_baseline(stack, quantile):
    """
    Finds quantile value for pixel and uses it as baseline.
    Parameters
    ----------
    stack : np.array 3D
        First dimension should encode time.
        Second and third dimension are for space.
    quantile : float
        Value betweeen 0 and 1.
    Returns
    -------
    baseline_img : np.array 3D
        Baseline image.
    """
    baseline_img = np.zeros(stack.shape[1:])
    for i in range(stack.shape[1]):
        for j in range(stack.shape[2]):
            baseline_img[i, j] = np.quantile(stack[:, i, j], quantile)
    return baseline_img

def _quantile_filt_baseline(stack, quantile, n=10):
    stack_filt = convolve(stack, np.expand_dims(np.ones(n), axis=(1,2)), mode="valid")
    return np.quantile(stack_filt, q=quantile, axis=0)

def find_dff_mask(baseline, otsu_frac=0.4, kernel=np.ones((20,20)), sigma=0, crop=None):  # 0.4, 30, 30, 10
    baseline = get_stack(baseline)
    baseline_filt = gaussian_filter(baseline, sigma=(sigma, sigma))
    mask = binary_closing(baseline_filt > otsu_frac*threshold_otsu(baseline_filt), selem=kernel)
    mask = crop_img(mask, crop)
    return mask

def compute_dff_from_stack(stack, baseline_blur=10, baseline_med_filt=1, blur_pre=True, baseline_mode="convolve", # slow alternative: "quantile"
                           baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                           use_crop=False, manual_add_to_crop=20, min_baseline=None,
                           dff_blur=0, dff_out_dir=None, return_stack=True):
    # load from path in case stack is a path. if numpy array, then just continue
    stack = get_stack(stack)
    N_frames, N_y, N_x = stack.shape

    dff_baseline = find_dff_baseline(stack, baseline_blur=baseline_blur, 
                                     baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                     baseline_mode=baseline_mode, baseline_length=baseline_length,
                                     baseline_quantile=baseline_quantile, baseline_dir=baseline_dir,
                                     min_baseline=min_baseline)
        
    # 3. compute cropping indices or use the ones supplied externally
    if (isinstance(use_crop, list) or isinstance(use_crop, tuple)) and len(use_crop) == 4:
        x_min, x_max, y_min, y_max = use_crop
    elif isinstance(use_crop, bool) and use_crop:
        z_projected = np.std(stack, axis=0)
        threshold_value = threshold_otsu(z_projected)
        mask = z_projected > threshold_value
        mask = binary_opening(mask, selem=np.ones((3, 3)))
        idx = np.where(mask)
        y_min = np.maximum(np.min(idx[0]) - manual_add_to_crop, 0)
        y_max = np.minimum(np.max(idx[0]) + manual_add_to_crop, stack.shape[1])
        x_min = np.maximum(np.min(idx[1]) - manual_add_to_crop, 0)
        x_max = np.minimum(np.max(idx[1]) + manual_add_to_crop, stack.shape[2])
    else:
        x_min = 0
        x_max = N_x
        y_min = 0
        y_max = N_y
    
    #4. apply cropping
    stack = stack[:, y_min : y_max, x_min : x_max]
    dff_baseline = dff_baseline[y_min : y_max, x_min : x_max]
    
    # 5. compute dff
    # this also applies a median filter with (3,3,3) kernel 
    dff = _compute_dff(stack,dff_baseline, apply_filter=True)
    
    # 6. post-process dff
    dff = gaussian_filter(dff, (0, dff_blur, dff_blur)) if dff_blur else dff

    if dff_out_dir is not None:
        utils2p.save_img(dff_out_dir, dff)

    if return_stack:
        return dff
    else:
        return None

def find_dff_baseline(stack, baseline_blur=10, baseline_med_filt=1, blur_pre=True, baseline_mode="convolve", # slow alternative: "quantile"
                      baseline_length=10, baseline_quantile=0.05, baseline_dir=None, min_baseline=None):
    # load from path in case stack is a path. if numpy array, then just continue
    stack = get_stack(stack)
    N_frames, N_y, N_x = stack.shape

    # 0. clip stack at 0
    stack = np.clip(stack, 0, None)
    # 1. blur stack if required
    stack_blurred = gaussian_filter(medfilt(stack, [1, baseline_med_filt, baseline_med_filt]), (0, baseline_blur, baseline_blur)) if baseline_blur and blur_pre else stack
    
    # 2. compute baseline
    if baseline_mode == "convolve":
        dff_baseline = _find_pixel_wise_baseline(stack_blurred, n=baseline_length)
    elif baseline_mode == "quantile":
        # dff_baseline = _quantile_baseline(stack_blurred, baseline_quantile)
        dff_baseline = _quantile_filt_baseline(stack_blurred, baseline_quantile, baseline_length)
    elif isinstance(baseline_mode, np.ndarray) and baseline_mode.shape == (N_y, N_x):
        dff_baseline = baseline_mode
    elif baseline_mode == "fromfile":
        dff_baseline = utils2p.load_img(baseline_dir)
        assert dff_baseline.shape == (N_y, N_x)
    else:
        raise NotImplementedError("baseline should be either 'convolve', 'quantile', or 'fromfile.")

    if not blur_pre and baseline_blur:
        dff_baseline = gaussian_filter(medfilt(dff_baseline, [baseline_med_filt, baseline_med_filt]), (baseline_blur, baseline_blur))
    
    if min_baseline is not None:
        dff_baseline[dff_baseline <= min_baseline] = 0  # set to 0 because then the dff will be set to zero throughout by compute_dff

    if baseline_dir is not None and baseline_mode != "fromfile":
        utils2p.save_img(baseline_dir, dff_baseline)

    return dff_baseline

def find_dff_baseline_multi_stack(stacks, baseline_blur=10, baseline_med_filt=1, blur_pre=True, baseline_mode="convolve", # slow alternative: "quantile"
                                  baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                                  return_multiple_baselines=False, min_baseline=None):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]

    # 0. clip stack at 0
    stacks = [np.clip(stack, 0, None) for stack in stacks]
    # 1. blur stack if required
    stacks_blurred = [gaussian_filter(medfilt(stack, [1, baseline_med_filt, baseline_med_filt]), (0, baseline_blur, baseline_blur)) if baseline_blur and blur_pre else stack 
                      for stack in stacks]

    # 2. concatenate stacks
    # stacks_cat = np.concatenate(stacks_blurred, axis=0)
    _, N_y, N_x = stacks[0].shape

    if baseline_mode == "convolve":
        dff_baseline = np.array([_find_pixel_wise_baseline(stack, n=baseline_length) for stack in stacks_blurred])
        if not return_multiple_baselines:
            dff_baseline = np.min(dff_baseline, axis=0)
    elif baseline_mode == "quantile":
        # dff_baseline = np.array([_quantile_baseline(stack, baseline_quantile) for stack in stacks_blurred])
        dff_baseline = np.array([_quantile_filt_baseline(stack, baseline_quantile, baseline_length) for stack in stacks_blurred])
        if not return_multiple_baselines:
            dff_baseline = np.min(dff_baseline, axis=0)
    elif isinstance(baseline_mode, np.ndarray) and baseline_mode.shape == (N_y, N_x):
        dff_baseline = baseline_mode
    elif baseline_mode == "fromfile":
        dff_baseline = utils2p.load_img(baseline_dir)
        assert dff_baseline.shape == (N_y, N_x)
    else:
        raise NotImplementedError("baseline should be either 'convolve', 'quantile', or 'fromfile.")

    if not blur_pre and baseline_blur:
        dff_baseline = gaussian_filter(medfilt(dff_baseline, [baseline_med_filt, baseline_med_filt]), (baseline_blur, baseline_blur))

    if min_baseline is not None:
        dff_baseline[dff_baseline <= min_baseline] = 0

    if baseline_dir is not None and baseline_mode != "fromfile":
        utils2p.save_img(baseline_dir, dff_baseline)

    return dff_baseline

def find_dff_baseline_multi_stack_load_single(stacks, individual_baseline_dirs,
                                              baseline_blur=10, baseline_med_filt=1,
                                              blur_pre=True, 
                                              baseline_mode="convolve", # slow alternative: "quantile"
                                              baseline_length=10, baseline_quantile=0.05, 
                                              baseline_dir=None, min_baseline=None):
    if not isinstance(stacks, list):
        stacks = [stacks]
    if not isinstance(individual_baseline_dirs, list):
        individual_baseline_dirs = [individual_baseline_dirs]
    assert len(stacks) == len(individual_baseline_dirs)

    baselines = [find_dff_baseline(stack=stack, baseline_blur=baseline_blur, 
                                   baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                   baseline_mode=baseline_mode, baseline_length=baseline_length,
                                   baseline_quantile=baseline_quantile,
                                   baseline_dir=trial_baseline_dir)
                 for i_trial, (stack, trial_baseline_dir) 
                 in enumerate(zip(stacks, individual_baseline_dirs))]

    dff_baseline = np.min(np.array(baselines), axis=0)
    if min_baseline is not None:
        dff_baseline[dff_baseline <= min_baseline] = 0

    if baseline_dir is not None:
        utils2p.save_img(baseline_dir, dff_baseline)

    return dff_baseline
                
def find_dff_crop_multi_stack(stacks, baseline_blur=10, manual_add_to_crop=20):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]

    # 1. blur stack if required TODO: potentially speed up by not requiring blurring for crop detection
    stacks_blurred = [gaussian_filter(stack, (0, baseline_blur, baseline_blur)) if baseline_blur else stack 
                      for stack in stacks]

    # 2. concatenate stacks
    stacks_cat = np.concatenate(stacks_blurred, axis=0)

    N_frames, N_y, N_x = stacks_cat.shape
    z_projected = np.std(stacks_cat, axis=0)
    threshold_value = threshold_otsu(z_projected)
    mask = z_projected > threshold_value
    mask = binary_opening(mask, selem=np.ones((3, 3)))
    idx = np.where(mask)
    y_min = np.maximum(np.min(idx[0]) - manual_add_to_crop, 0)
    y_max = np.minimum(np.max(idx[0]) + manual_add_to_crop, stacks_cat.shape[1])
    x_min = np.maximum(np.min(idx[1]) - manual_add_to_crop, 0)
    x_max = np.minimum(np.max(idx[1]) + manual_add_to_crop, stacks_cat.shape[2])

    return (x_min, x_max, y_min, y_max)

def compute_dff_multi_stack(stacks, baseline_blur=10, baseline_med_filt=1, blur_pre=True, baseline_mode="convolve", # slow alternative: "quantile"
                           baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                           use_crop=False, manual_add_to_crop=20, min_baseline=None,
                           dff_blur=0, dff_out_dirs=None, return_stacks=True):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]
    dff_baseline = find_dff_baseline_multi_stack(stacks, baseline_blur=baseline_blur, 
                                                 baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                                 baseline_mode=baseline_mode,
                                                 baseline_length=baseline_length, baseline_quantile=baseline_quantile,
                                                 baseline_dir=baseline_dir, min_baseline=min_baseline)

    if (isinstance(use_crop, list) or isinstance(use_crop, tuple)) and len(use_crop) == 4:
        crop_all_stacks = use_crop
    elif isinstance(use_crop, bool) and use_crop:
        crop_all_stacks = find_dff_crop_multi_stack(stacks, baseline_blur=baseline_blur, manual_add_to_crop=manual_add_to_crop) if use_crop else use_crop
    else:
        crop_all_stacks = use_crop
        
    dff_out_dirs = [None for stack in stacks] if dff_out_dirs is None else dff_out_dirs
    assert len(dff_out_dirs) == len(stacks)

    dffs = [compute_dff_from_stack(stack=stack, baseline_blur=0, baseline_mode=dff_baseline, baseline_dir=None, 
                                   use_crop=crop_all_stacks, dff_blur=dff_blur, dff_out_dir=dff_out_dir, return_stack=return_stacks) 
            for stack, dff_out_dir in zip(stacks, dff_out_dirs)]

    return dffs


if __name__ == "__main__":
    COMPUTE = False
    VIDEOS = False
    VIDEOS_2P = False
    ANALYSE = False
    MULTIBASELINE = False
    DENOISED = True

    date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    all_trial_dirs = load.get_trials_from_fly(fly_dirs)
    for fly_dir, trial_dirs in zip(fly_dirs, all_trial_dirs):
        fly_processed_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER)
        processed_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER) for trial_dir in trial_dirs]
        green_warped_dirs = [os.path.join(processed_dir, "green_com_warped.tif") for processed_dir in processed_dirs]  # green_com_warped
        red_warped_dirs = [os.path.join(processed_dir, "red_com_warped.tif") for processed_dir in processed_dirs] 
        baseline_dir = os.path.join(fly_processed_dir, "dff_baseline_com_warped_003_005_007_010.tif")  # dff_baseline_com_warped_003_005_007_010
        dff_out_dirs = [os.path.join(processed_dir, "dff_com_warped.tif") for processed_dir in processed_dirs]  # dff_com_warped
        dff_video_dirs = [os.path.join(processed_dir, "dff_com_warped.mp4") for processed_dir in processed_dirs]  # dff_com_warped

        ext_crops = (74, 736-74, 48, 480-48)  # (74, 736-74, 48+38, 480-25)  # crop 10 % of the image on each side  (74, 736-74, 48, 480-48)

        if COMPUTE:
            print("computing dff")
            dffs = compute_dff_multi_stack(stacks=green_warped_dirs, baseline_blur=10, baseline_mode="convolve",  # baselin_blur=3
                                        baseline_length=10, baseline_dir=baseline_dir, 
                                        use_crop=ext_crops, manual_add_to_crop=20, dff_blur=0, 
                                        dff_out_dirs=dff_out_dirs, return_stacks=True)
        else:
            dffs = [get_stack(dff_out_dir) for dff_out_dir in dff_out_dirs]
        
        if VIDEOS:
            print("making videos")
            _ = [make_video_dff(dff, processed_dir, video_name="dff_com_warped", trial_dir=trial_dir, vmin=-10)  # dff_com_warped
                for dff, trial_dir, processed_dir in zip(dffs, trial_dirs, processed_dirs)]

            make_multiple_video_dff(dffs, fly_processed_dir, video_name="dff_com_warped_003_005_007_010", trial_dir=trial_dirs[0], vmin=-10)  # dff_com_warped_003_005_007_010

        if VIDEOS_2P:
            print("making 2p videos")
            make_multiple_video_2p(greens=green_warped_dirs, out_dir=fly_processed_dir, video_name="green_warped_003_005_007_010", reds=None, trial_dir=trial_dirs[0])
        if ANALYSE:
            dff_above_thres = [np.sum(dff > 50, axis=(1,2)) for dff in dffs]
            dff_names = [trial_dir[-3:] for trial_dir in trial_dirs]
            colors = ["b", "g", "m", "r"]
            fig, axs = plt.subplots(2,1, figsize=(10,10))
            _ = [axs[0].plot(gaussian_filter1d(dff_above_thre, 3), color, label=dff_name) for dff_above_thre, dff_name, color in zip(dff_above_thres, dff_names, colors)]
            axs[0].set_xlabel("samples")
            axs[0].set_ylabel("# of pixels above $\Delta F/F=50$")
            axs[0].legend()

            dff_names2 = [dff_name + ": {:5.1f} +- {:.1f}".format(np.mean(dff_above_thre), np.std(dff_above_thre)) for dff_above_thre, dff_name in zip(dff_above_thres, dff_names)]
            axs[1].hist(dff_above_thres, bins=20, label=dff_names2, color=colors)
            axs[1].set_xlabel("# of pixels above $\Delta F/F=50$")
            axs[1].set_ylabel("# of frames")
            axs[1].legend()
            plt.savefig(os.path.join(fly_processed_dir, "mean_dFF_warped.png"))

            active_neurons = [np.quantile(dff, 0.95, axis=0) for dff in dffs]
            differences = np.array([active_neuron - active_neurons[0] for active_neuron in active_neurons])
            relatives = differences / np.clip(active_neurons[0], 10, None)
            # clim = np.max([np.abs(np.quantile(differences, 0.95)), np.abs(np.quantile(differences, 0.05))])

            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,7))
            for i_t, (dff_name, active_neuron, difference, relative) in enumerate(zip(dff_names, active_neurons, differences, relatives)):
                axs[0, i_t].imshow(active_neuron, clim=[0, 150], cmap=plt.cm.jet)
                axs[0, i_t].set_title(dff_name + " 95% quantile")
                axs[1, i_t].imshow(difference, clim=[-50, 50], cmap=plt.cm.seismic)
                axs[1, i_t].set_title("difference to first trial")
                # axs[2, i_t].imshow(difference, clim=[-2, 2], cmap=plt.cm.seismic)
                # axs[2, i_t].set_title("relative difference to first trial")
                # fig.suptitle("clim = {:.2f}".format(clim))
            plt.savefig(os.path.join(fly_processed_dir, "max_dFF_trial_difference.png"))
            
        if MULTIBASELINE:
            baseline_dir = os.path.join(fly_processed_dir, "dff_baseline_warped_003_005_007_010_4x.tif")
            dff_baselines = find_dff_baseline_multi_stack(stacks=green_warped_dirs, baseline_blur=0, baseline_mode="convolve", baseline_length=10,
                                                          baseline_dir=baseline_dir, return_multiple_baselines=True)

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
            cmap = plt.cm.jet
            clim = [0, np.quantile(dff_baselines, 0.995)]
            baseline_names = [trial_dir[-3:] for trial_dir in trial_dirs]
            for i_b, dff_baseline in enumerate(dff_baselines):
                ax = axs[np.floor(i_b/2).astype(np.int), i_b%2]
                ax.imshow(dff_baseline, clim=clim, cmap=cmap)
                ax.set_title(baseline_names[i_b])

            minind_image = np.argmin(dff_baselines, axis=0)
            axs[2,0].imshow(minind_image,cmap=plt.cm.binary)
            axs[2,0].set_title("index of minimum baseline")
            axs[2,1].axis("off")
            fig.suptitle("$\Delta F/F baselines$")
            plt.savefig(os.path.join(fly_processed_dir, "dff_baseline_warped_003_005_007_010_4x_blur0.png"))
        
    if DENOISED:
        base_dir = "/home/jbraun/bin/deepinterpolation/sample_data"
        greens = [get_stack(os.path.join(base_dir, "longterm_003_crop.tif"))[30:-30],
                  get_stack(os.path.join(base_dir, "denoised_longterm_003_crop_out.tif")),
                  get_stack(os.path.join(base_dir, "denoised_halves_longterm_003_crop_out.tif"))]
        baseline_blurs = [10, 1, 1]
        dff_out_dirs = [os.path.join(base_dir, "dff_raw.tif"),
                        os.path.join(base_dir, "dff_denoised.tif"),
                        os.path.join(base_dir, "dff_denoised_halves.tif")]
        dffs = [compute_dff_from_stack(green, baseline_blur=baseline_blur, dff_out_dir=dff_out_dir, use_crop=False) 
                for green, baseline_blur, dff_out_dir in zip(greens, baseline_blurs, dff_out_dirs)]
        make_multiple_video_dff(dffs, out_dir=base_dir, video_name="compare_dff", frame_rate=8, vmin=0)

        pass
