# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import sys, os.path
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import nely_suite

FILE_PATH = os.path.realpath(__file__)
LONGTERM_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import get_stack
from longterm import load
from longterm.plot.videos import make_video_dff, make_multiple_video_dff, make_multiple_video_2p

def compute_dff_from_stack(stack, baseline_blur=3, baseline_med_filt=3, blur_pre=False, baseline_mode="convolve", # slow alternative: "quantile"
                           baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                           use_crop=True, manual_add_to_crop=20,
                           dff_blur=0, dff_out_dir=None, return_stack=True):
    # load from path in case stack is a path. if numpy array, then just continue
    stack = get_stack(stack)
    N_frames, N_y, N_x = stack.shape

    dff_baseline = find_dff_baseline(stack, baseline_blur=baseline_blur, 
                                     baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                     baseline_mode=baseline_mode, baseline_length=baseline_length,
                                     baseline_quantile=baseline_quantile, baseline_dir=baseline_dir)
        
    # 3. compute cropping indices or use the ones supplied externally
    if (isinstance(use_crop, list) or isinstance(use_crop, tuple)) and len(use_crop) == 4:
        x_min, x_max, y_min, y_max = use_crop
    elif isinstance(use_crop, bool) and use_crop:
        mask = nely_suite.analysis.background_mask(stack, z_projection="std", threshold="otsu")
        idx = np.where(mask)
        y_min = np.maximum(np.min(idx[0]) - manual_add_to_crop, 0)
        y_max = np.minimum(np.max(idx[0]) + manual_add_to_crop, stack.shape[1] - 1)
        x_min = np.maximum(np.min(idx[1]) - manual_add_to_crop, 0)
        x_max = np.minimum(np.max(idx[1]) + manual_add_to_crop, stack.shape[2] - 1)
    else:
        x_min = 0
        x_max = N_x - 1
        y_min = 0
        y_max = N_y - 1
    
    #4. apply cropping
    stack = nely_suite.crop_stack(stack, x_min, x_max, y_min, y_max)
    dff_baseline = dff_baseline[y_min : y_max + 1, x_min : x_max + 1]
    
    # 5. compute dff
    # this also applies a median filter with (3,3,3) kernel 
    # and ignores the areas set to 0 by motion correction
    dff = nely_suite.calculate_dff(stack,
                                   dff_baseline, 
                                   apply_filter=True, occlusion_handling=True)
    
    # 6. post-process dff
    dff = gaussian_filter(dff, (0, dff_blur, dff_blur)) if dff_blur else dff

    if dff_out_dir is not None:
        nely_suite.save_img(dff_out_dir, dff)

    if return_stack:
        return dff
    else:
        return None

def find_dff_baseline(stack, baseline_blur=3, baseline_med_filt=3, blur_pre=False, baseline_mode="convolve", # slow alternative: "quantile"
                      baseline_length=10, baseline_quantile=0.05, baseline_dir=None):
    # load from path in case stack is a path. if numpy array, then just continue
    stack = get_stack(stack)
    N_frames, N_y, N_x = stack.shape

    # 1. blur stack if required
    stack_blurred = gaussian_filter(medfilt(stack, [0, baseline_med_filt, baseline_med_filt]), (0, baseline_blur, baseline_blur)) if baseline_blur and blur_pre else stack
    
    # 2. compute baseline
    if baseline_mode == "convolve":
        dff_baseline = nely_suite.find_pixel_wise_baseline(stack_blurred, n=baseline_length)
    elif baseline_mode == "quantile":
        dff_baseline = nely_suite.quantile_baseline(stack_blurred, baseline_quantile)
    elif isinstance(baseline_mode, np.ndarray) and baseline_mode.shape == (N_y, N_x):
        dff_baseline = baseline_mode
    elif baseline_mode == "fromfile":
        dff_baseline = nely_suite.load_img(baseline_dir)
        assert dff_baseline.shape == (N_y, N_x)
    else:
        raise(NotImplementedError)

    if not blur_pre and baseline_blur:
        dff_baseline = gaussian_filter(medfilt(dff_baseline, [baseline_med_filt, baseline_med_filt]), (baseline_blur, baseline_blur))
    
    dff_baseline[dff_baseline <= 0] = 0

    if baseline_dir is not None and baseline_mode != "fromfile":
        nely_suite.save_img(baseline_dir, dff_baseline)

    return dff_baseline

def find_dff_baseline_multi_stack(stacks, baseline_blur=3, baseline_med_filt=3, blur_pre=False, baseline_mode="convolve", # slow alternative: "quantile"
                                  baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                                  return_multiple_baselines=False):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]

    # 1. blur stack if required
    stacks_blurred = [gaussian_filter(medfilt(stack, [0, baseline_med_filt, baseline_med_filt]), (0, baseline_blur, baseline_blur)) if baseline_blur and blur_pre else stack 
                      for stack in stacks]

    # 2. concatenate stacks
    # stacks_cat = np.concatenate(stacks_blurred, axis=0)
    _, N_y, N_x = stacks[0].shape

    if baseline_mode == "convolve":
        dff_baseline = np.array([nely_suite.find_pixel_wise_baseline(stack, n=baseline_length) for stack in stacks_blurred])
        if not return_multiple_baselines:
            dff_baseline = np.min(dff_baseline, axis=0)
    elif baseline_mode == "quantile":
        dff_baseline = np.array([nely_suite.quantile_baseline(stack, baseline_quantile) for stack in stacks_blurred])
        if not return_multiple_baselines:
            dff_baseline = np.min(dff_baseline, axis=0)
    elif isinstance(baseline_mode, np.ndarray) and baseline_mode.shape == (N_y, N_x):
        dff_baseline = baseline_mode
    elif baseline_mode == "fromfile":
        dff_baseline = nely_suite.load_img(baseline_dir)
        assert dff_baseline.shape == (N_y, N_x)
    else:
        raise(NotImplementedError)

    if not blur_pre and baseline_blur:
        dff_baseline = gaussian_filter(medfilt(dff_baseline, [baseline_med_filt, baseline_med_filt]), (baseline_blur, baseline_blur))

    dff_baseline[dff_baseline <= 0] = 0

    if baseline_dir is not None and baseline_mode != "fromfile":
        nely_suite.save_img(baseline_dir, dff_baseline)

    return dff_baseline

def find_dff_baseline_multi_stack_load_single(stacks, individual_baselin_dirs,
                                              baseline_blur=3, baseline_med_filt=3,
                                              blur_pre=False, 
                                              baseline_mode="convolve", # slow alternative: "quantile"
                                              baseline_length=10, baseline_quantile=0.05, 
                                              baseline_dir=None):
    if not isinstance(stacks, list):
        stacks = [stacks]
    if not isinstance(individual_baselin_dirs, list):
        individual_baselin_dirs = [individual_baselin_dirs]
    assert len(stacks) == len(individual_baselin_dirs)

    baselines = [find_dff_baseline(stack=stack, baseline_blur=baseline_blur, 
                                   baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                   baseline_mode=baseline_mode, baseline_length=baseline_length,
                                   baseline_quantile=baseline_quantile,
                                   baseline_dir=trial_baseline_dir)
                 for i_trial, (stack, trial_baseline_dir) 
                 in enumerate(zip(stacks, individual_baselin_dirs))]

    dff_baseline = np.min(np.array(baselines), axis=0)
    dff_baseline[dff_baseline <= 0] = 0

    # TODO: check whether we want to blur again after taking the minimum
    # if not blur_pre and baseline_blur:
    #     dff_baseline = gaussian_filter(dff_baseline, (baseline_blur, baseline_blur))

    if baseline_dir is not None:
        nely_suite.save_img(baseline_dir, dff_baseline)

    return dff_baseline
                

def find_dff_crop_multi_stack(stacks, baseline_blur=0, manual_add_to_crop=20):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]

    # 1. blur stack if required TODO: potentially speed up by not requiring blurring for crop detection
    stacks_blurred = [gaussian_filter(stack, (0, baseline_blur, baseline_blur)) if baseline_blur else stack 
                      for stack in stacks]

    # 2. concatenate stacks
    stacks_cat = np.concatenate(stacks_blurred, axis=0)

    N_frames, N_y, N_x = stacks_cat.shape
    mask = nely_suite.analysis.background_mask(stacks_cat, z_projection="std", threshold="otsu")
    idx = np.where(mask)
    y_min = np.maximum(np.min(idx[0]) - manual_add_to_crop, 0)
    y_max = np.minimum(np.max(idx[0]) + manual_add_to_crop, stacks_cat.shape[1] - 1)
    x_min = np.maximum(np.min(idx[1]) - manual_add_to_crop, 0)
    x_max = np.minimum(np.max(idx[1]) + manual_add_to_crop, stacks_cat.shape[2] - 1)

    return (x_min, x_max, y_min, y_max)

def compute_dff_multi_stack(stacks, baseline_blur=3, baseline_med_filt=3, blur_pre=False, baseline_mode="convolve", # slow alternative: "quantile"
                           baseline_length=10, baseline_quantile=0.05, baseline_dir=None,
                           use_crop=True, manual_add_to_crop=20,
                           dff_blur=0, dff_out_dirs=None, return_stacks=True):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [get_stack(stack) for stack in stacks]
    dff_baseline = find_dff_baseline_multi_stack(stacks, baseline_blur=baseline_blur, 
                                                 baseline_med_filt=baseline_med_filt, blur_pre=blur_pre,
                                                 baseline_mode=baseline_mode,
                                                 baseline_length=baseline_length, baseline_quantile=baseline_quantile,
                                                 baseline_dir=baseline_dir)

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
