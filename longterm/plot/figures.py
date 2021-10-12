import os
import sys
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage.color import label2rgb
from PIL import ImageColor

FILE_PATH = os.path.realpath(__file__)
PLOT_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(PLOT_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)
OUTPUT_PATH = os.path.join(MODULE_PATH, "outputs")

from longterm import utils, load, rois, denoise
from longterm import high_caff_flies, high_caff_main_fly, low_caff_main_fly, sucr_main_fly
from longterm import plot as myplt

colors = [myplt.DARKBLUE, myplt.DARKBLUE_CONTRAST, myplt.DARKCYAN, myplt.DARKGREEN, myplt.DARKGREEN_CONTRAST,
          myplt.DARKYELLOW, myplt.DARKORANGE, myplt.DARKPINK, myplt.DARKRED, myplt.DARKPURPLE,
          myplt.DARKBROWN, myplt.DARKGRAY, myplt.BLACK]

def compute_rest_maps(fly, trials, correct=True):
    dffs = [utils.get_stack(fly.trials[trial].dff) for trial in trials]
    greens = [utils.get_stack(fly.trials[trial].green_raw) for trial in trials]
    green_denoiseds = [utils.get_stack(fly.trials[trial].green_denoised) for trial in trials]
    twop_dfs = [pd.read_pickle(fly.trials[trial].twop_df) for trial in trials]
    rests = [twop_df["rest"].values for twop_df in twop_dfs]  # rest rest_cleaned
    if correct:
        if not os.path.isfile(fly.correction):
            with open(fly.summary_dict, "rb") as f:
                summary_dict = pickle.load(f)
            green_mean = np.mean(summary_dict["green_means_raw"], axis=0)
            del summary_dict
            correction = denoise.get_illumination_correction(green_mean)
            with open(fly.correction, "wb") as f:
                summary_dict = pickle.dump(correction, f)
        else:
            with open(fly.correction, "rb") as f:
                correction = pickle.load(f)
        greens = [denoise.correct_illumination(stack, correction) for stack in greens]


    denoised_crop = (len(greens[0]) - len(rests[0])) // 2

    dff_means = [np.mean(dff[rest,:,:], axis=0) for dff, rest in zip(dffs, rests)]
    dff_maxs = [np.quantile(dff[rest,:,:], 0.995, axis=0) for dff, rest in zip(dffs, rests)]
    dff_mins = [np.quantile(dff[rest,:,:], 0.005, axis=0) for dff, rest in zip(dffs, rests)]
    green_means = [np.mean(green[denoised_crop:denoised_crop+len(rest)][rest,:,:], axis=0) for green, rest in zip(greens, rests)]
    green_maxs = [np.quantile(green[denoised_crop:denoised_crop+len(rest)][rest,:,:], 0.995, axis=0) for green, rest in zip(greens, rests)]
    green_mins = [np.quantile(green[denoised_crop:denoised_crop+len(rest)][rest,:,:], 0.005, axis=0) for green, rest in zip(greens, rests)]
    green_denoised_means = [np.mean(green[rest,:,:], axis=0) for green, rest in zip(green_denoiseds, rests)]
    green_denoised_maxs = [np.quantile(green[rest,:,:], 0.995, axis=0) 
                           for green, rest in zip(green_denoiseds, rests)]
    green_denoised_mins = [np.quantile(green[rest,:,:], 0.005, axis=0) 
                           for green, rest in zip(green_denoiseds, rests)]

    map_dict = {
        "dff_means": dff_means,
        "dff_mins": dff_mins,
        "dff_maxs": dff_maxs,
        "green_means": green_means,
        "green_mins": green_mins,
        "green_maxs": green_maxs,
        "green_denoised_means": green_denoised_means,
        "green_denoised_mins": green_denoised_mins,
        "green_denoised_maxs": green_denoised_maxs,
        "trial_names": [fly.trials[trial].name for trial in trials]
    }
    with open(fly.rest_maps, "wb") as f:
            pickle.dump(map_dict, f)
    return map_dict

def plot_dff_maps(map_dict, fly_dir):
    N_trials = len(map_dict["means_rest"])
    fig, axs = plt.subplots(N_trials, 4, figsize=(16,N_trials*3))
    clim = [0, np.maximum(1,np.mean(map_dict["dff_max"]))]  # np.mean(map_dict["dff_min"])
    ims = ["means_walk", "means_rest", "means_walk_norm", "means_rest_norm"]
    clims = [clim, clim, [0,1], [0,1]]
    titles = ["mean: walk", "mean: rest", "normalised mean: walk", "normalised mean: rest"]
    cmaps = [plt.cm.get_cmap("jet"),plt.cm.get_cmap("jet"), plt.cm.get_cmap("viridis"), plt.cm.get_cmap("viridis")]
    for i_trial, axs_row in enumerate(axs):
        for i_col, ax in enumerate(axs_row):
            im = map_dict[ims[i_col]][i_trial]
            if len(im):
                ax.imshow(im, clim=clims[i_col], cmap=cmaps[i_col])
            else:
                ax.axis("off")
            title = titles[i_col]
            if i_col == 0:
                title = map_dict["trial_names"][i_trial] + " " + title
            ax.set_title(title)
    fig.suptitle(fly_dir)
    fig.tight_layout()
    return fig

def make_dff_maps(fly, trials, axs, overwrite=False, data="dff", set_title=True, i_cbar_ax=2, crop=True):
    _ = [ax.axis("off") for ax in axs]
    if not os.path.isfile(fly.rest_maps) or overwrite:
        map_dict = compute_rest_maps(fly, trials)
    else:
        with open(fly.rest_maps, "rb") as f:
            map_dict = pickle.load(f)
    mask = utils.get_stack(fly.mask_fine) > 0

    im_max = np.maximum(1, np.mean(map_dict[data+"_maxs"], where=mask))
    im_min = 0  # np.maximum(0, np.mean(map_dict[data+"_mins"], where=mask))

    ims = map_dict[data+"_means"]

    if isinstance(crop, bool) and crop:
        to_crop = [35, 50]
        
    elif isinstance(crop, list):
        to_crop = crop
        crop = True

    if data == "dff":
        im_max = 300
        data = r"$\Delta F/F$ (%)"
    elif data == "green":
        im_max = 2000
        im_min = 500
        data = "fluorescence (a.u.)"
    elif data == "green_denoised":
        pass
    titles = ["before feeding", "right after feeding", "long after feeding"]
    for i_ax, (ax, im, title) in enumerate(zip(axs, ims, titles)):
        # if data != "green":
        im[np.logical_not(mask)] = None
        if crop:
            im = im[to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]]
        im = ax.imshow(im, cmap=plt.cm.get_cmap("jet"), clim=[im_min, im_max])
        if set_title:
            ax.set_title(title)
        if i_ax == i_cbar_ax:
            cbar = ax.figure.colorbar(im, orientation="vertical", ax=ax, aspect=20, shrink=0.8)
        # cbar.set_ticks()
            cbar.set_label(data.replace("_", " "))
            cbar.outline.set_visible(False)
    
def make_detailled_wave_plots(fly, axs, i_trial, overwrite=False, crop=True, title=None, legend=True):
    # 0. load data
    # 0.1 summary image data
    if os.path.isfile(fly.raw_std) and not overwrite:
        ref_img = utils.get_stack(fly.raw_std)
    else:
        with open(fly.summary_dict, "rb") as f:
            summary_dict = pickle.load(f)
        ref_img = np.std(summary_dict["green_stds_raw"], axis=0)
        utils.save_stack(fly.raw_std, ref_img)
    wave_details_file = fly.wave_details[:-4] + f"_{i_trial}.pkl"
    if os.path.isfile(wave_details_file) and not overwrite:
        with open(wave_details_file, "rb") as f:
            wave_details = pickle.load(f)
    else:
        wave_details = {}
        wave_details["ref_img"] = ref_img

        if isinstance(crop, bool) and crop:
            to_crop = [35, 50]
            
        elif isinstance(crop, list):
            to_crop = crop
            crop = True
        if crop:
            ref_img = ref_img[to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]]
        # 0.2 raw data
        green = utils.get_stack(fly.trials[i_trial].green_denoised)
        if crop:
            green = green[:,to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]]
        # 0.3 over-all mask
        mask = utils.get_stack(fly.mask_coarse) > 0
        if crop:
            mask = mask[to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]]
        mask_fine = utils.get_stack(fly.mask_fine) > 0
        if crop:
            mask_fine = mask_fine[to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]]
        # 0.4 roi masks
        roi_masks = [utils.get_stack(this_mask) > 0 for this_mask in fly.wave_masks]
        if crop:
            roi_masks = [this_mask[to_crop[0]:-to_crop[0], to_crop[1]:-to_crop[1]] for this_mask in roi_masks]
        # roi_colors = [colors[i_col] for i_col in [0, 3, 3, 6, 8, 8]]
        # roi_labels = ["top", "left", "right", "bottom", "giant fiber left", "giant fiber right"]
        mask_dorsal = roi_masks[0]
        mask_lateral = (roi_masks[1] + roi_masks[2]) > 0
        mask_ventral = roi_masks[3]
        mask_giantfiber = (roi_masks[4] + roi_masks[5]) > 0
        roi_summary_masks = [mask_dorsal, mask_lateral, mask_ventral, mask_giantfiber]
        roi_summary_labels = ["dorsal", "lateral", "ventral", "giant fiber"]
        roi_summary_colors_rgb = [np.array(ImageColor.getrgb(colors[i_col]))/255 for i_col in [0, 3, 6, 8]]
        roi_summary_img = mask_dorsal.astype(int) + 2*mask_lateral.astype(int) + 3*mask_ventral.astype(int) + 4*mask_giantfiber.astype(int)
        # 1. first plot: overview of CC with regions of interest
        ref_img = np.clip(ref_img/np.quantile(ref_img, 0.99), a_min=0, a_max=1)
        image_label_overlay = label2rgb(roi_summary_img, image=ref_img, bg_label=0, colors=roi_summary_colors_rgb,
                                        image_alpha=1, alpha=0.3)

        wave_details["roi_summary_masks"] = roi_summary_masks
        wave_details["roi_summary_labels"] = roi_summary_labels
        wave_details["roi_summary_colors_rgb"] = roi_summary_colors_rgb
        wave_details["roi_summary_img"] = roi_summary_img
        wave_details["image_label_overlay"] = image_label_overlay

        def lp_filter_and_normalise_roi(mask, stack):
            signal = np.sum(stack*mask, axis=(1,2)) / np.sum(mask)
            signal = gaussian_filter1d(signal, sigma=3)
            return (signal - signal.min()) / (signal.max()-signal.min())

        # 2. second plot: waves of regions of interest
        global_signal = lp_filter_and_normalise_roi(mask, green)

        i_global_max = np.argmax(global_signal, axis=0)
        t_range_long = [-50,75]
        i_range_long = [i_global_max + int(fly.fs*t_range_long[0]), i_global_max + int(fly.fs*t_range_long[1])]

        t = (np.arange(len(global_signal)) - i_global_max) / fly.fs

        roi_summary_signals = [lp_filter_and_normalise_roi(this_mask, green) for this_mask in roi_summary_masks]
        wave_details["t"] = t
        wave_details["i_range_long"] = i_range_long
        wave_details["t_range_long"] = t_range_long
        wave_details["global_signal"] = global_signal
        wave_details["i_global_max"] = i_global_max
        wave_details["roi_summary_signals"] = roi_summary_signals
       

        # 3. compute timing in CC map
        green_filt = gaussian_filter1d(green, sigma=10, axis=0)
        t_range_short = [-15,15]
        t_range_show = [-10, 10]
        i_range_short = [i_global_max + int(fly.fs*t_range_short[0]), i_global_max + int(fly.fs*t_range_short[1])]
        green_filt_crop = green_filt[i_range_short[0]:i_range_short[1]]
        green_filt_crop = gaussian_filter(green_filt_crop, sigma=(0,1,1))
        # find time point of maximum fluorescence around wave
        green_argmax = np.argmax(green_filt_crop, axis=0).astype(float)
        # convert to time in seconds relative to wave
        green_argmax = green_argmax/fly.fs + t_range_short[0]
        # apply cervical connective mask
        try:
            green_argmax[np.logical_not(mask_fine)] = None
        except:
            green_argmax[np.logical_not(mask)] = None
        wave_details["green_argmax"] = green_argmax
        wave_details["t_range_show"] = t_range_show

        with open(wave_details_file, "wb") as f:
            pickle.dump(wave_details, f)


    ## do the actual plotting
    # subplot 0: CC map + regions of interest
    try:
        axs[0].imshow(wave_details["image_label_overlay"])
    except:
        axs[0].imshow(wave_details["ref_img"], cmap=plt.cm.get_cmap("gray"), 
                     clim=[0, np.quantile(wave_details["ref_img"], 0.99)])
    if title == "auto":
        title = str(fly.date) + " fly " + str(fly.i_flyonday) + " " + fly.trials[i_trial].name
        axs[0].set_title(title)
    elif title is not None:
        axs[0].set_title(title)
    axs[0].axis("off")
    # subplot 1: time traces
    # axs[1].plot(t,global_signal,color=colors[-1],linewidth=4,alpha=0.3,label="entire connective")
    _ = [axs[1].plot(wave_details["t"], s, color=c, linewidth=2, alpha=1, label=l) for
            s,c,l in zip(wave_details["roi_summary_signals"],
                         wave_details["roi_summary_colors_rgb"],
                         wave_details["roi_summary_labels"])]
    axs[1].set_ylim(-0.02,1.02)
    axs[1].set_ylabel("normalised fluorescence")
    axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[1].set_xlim(wave_details["t_range_long"])
    axs[1].set_xlabel("t (s)")
    axs[1].set_xticks([-50, -25, 0, 25, 50, 75])  # -50, , 100
    # axs[1].legend(frameon=False)
    if legend:
        for i_t, (t, c) in enumerate(zip(wave_details["roi_summary_labels"],
                                         wave_details["roi_summary_colors_rgb"])):
            axs[1].text(x=50, y=0.9-i_t*0.08, s=t, color=c)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    # subplot 2: timing map
    im = axs[2].imshow(wave_details["green_argmax"], cmap=plt.cm.get_cmap("turbo"),  # turbo, seismic
                       clim=wave_details["t_range_show"])
    axs[2].axis("off")

    cbar = axs[2].figure.colorbar(im, orientation="vertical", ax=axs[2],  aspect=20, shrink=0.8)  # cax=axs[3],
    cbar.set_ticks([wave_details["t_range_show"][0], 0.5*wave_details["t_range_show"][0], 0, 
                    0.5*wave_details["t_range_show"][1], wave_details["t_range_show"][1]])
    cbar.set_label("peak time (s)")
    cbar.outline.set_visible(False)

def make_trial_wave_plot(fly, ax, ylim=None, overwrite=False, legend=False, shift=1, normalise=0.99):
    mask = utils.get_stack(fly.mask_coarse) > 0
    N_pixels = np.sum(mask)
    green_means = []
    if os.path.isfile(fly.trials_mean_dff) and not overwrite:
        print("loading pre-computed mean fluorescence for fly ", fly.dir)
        with open(fly.trials_mean_dff, "rb") as f:
            dffs = pickle.load(f)
    else:
        print("loading fluorescence data for fly ", fly.dir)
        for trial in tqdm(fly.trials):
            green = utils.get_stack(trial.green_denoised)
            green_mean = np.sum(green*mask, axis=(1,2))/N_pixels
            green_means.append(green_mean)
        green_means = np.array(green_means).T
        
        _, f0s = rois.get_dff_from_traces(green_means, return_f0=True, length_baseline=50, f0_min=10)
        f0 = f0s.min()
        dffs = 100*(green_means - f0) / f0
        with open(fly.trials_mean_dff, "wb") as f:
            pickle.dump(dffs, f)

    if normalise:
        low = np.quantile(dffs[:,:2], (1-normalise)/2)
        high = np.quantile(dffs[:,:2], (1+normalise)/2)
        dffs = (dffs - low) / (high - low)

    trial_colors = [colors[0], colors[0], colors[3]] + [colors[6]]*7 + [colors[8]]*2
    trial_alphas = [0.5, 0.8, 0.8] + list(np.linspace(0.5, 0.8, 7)) + [0.7, 1]

    t = np.arange(np.max(dffs.shape)) / fly.fs
    for i_l, (line, c, a, name) in enumerate(zip(dffs.T, trial_colors, trial_alphas, fly.trial_names)):
        ax.plot(t, line+i_l*shift, color=c, alpha=a, label=name)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    yticks = np.arange(0,ylim[-1],shift)
    ax.set_yticks(yticks)
    ax.set_yticklabels([0,shift]+["" for _ in yticks[2:]])

    dist = np.maximum(shift, 1.5)
    
    if legend:
        texts = ["before feeding", "during feeding", "<25 min after feeding", ">25 min after feeding"]
        text_colors = [colors[0], colors[3], colors[6], colors[8]]
        for i_t, (t, c) in enumerate(zip(texts, text_colors)):
            ax.text(x=0, y=ylim[-1]-4*dist+i_t*dist, s=t, color=c)

    # ax.legend(frameon=False)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("normalized fluorescence")  # r"$\Delta F/F$ (%)")  # mean Fluorescence (a.u.)")
    ax.set_title(fly.paper_condition)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax.get_ylim()

def main_figure(map_type="dff"):
    fig = plt.figure(figsize=(15,10))  # 15
    layout = """
    AAAAABBBBBCCCCCX
    GGGGGHHHHHIIIIII
    DDDDDDEEEEFFFFFF
    """
    # JJJJJKKKKKLLLLL
    # MMMMMNNNNNOOOOO
    
    ax_dict = fig.subplot_mosaic(layout)

    # row 1: trial wave plots
    ylim = make_trial_wave_plot(high_caff_main_fly, ax_dict["C"], legend=False)  # axs[0,0])
    _ = make_trial_wave_plot(low_caff_main_fly, ylim=ylim, ax=ax_dict["B"], legend=False)  # axs[0,1],
    _ = make_trial_wave_plot(sucr_main_fly, ylim=ylim, ax=ax_dict["A"], legend=True)  # axs[0,2],

    # row 2: wave detailled plot
    make_detailled_wave_plots(high_caff_main_fly, crop=True,
                              axs=[ax_dict[k] for k in ["E", "D", "F"]], i_trial=-2)  #axs[1,:]

    # row 3-5: dff maps for high caff, low caff, sucrose
    make_dff_maps(high_caff_main_fly, trials=[1,3,-2], data=map_type, set_title=True,
                  axs=[ax_dict[k] for k in ["G", "H", "I"]], crop=[35,50])  # axs[1:,0])
    """
    make_dff_maps(low_caff_main_fly, trials=[1,3,-1], data=map_type, set_title=False,
                  axs=[ax_dict[k] for k in ["H", "K", "N"]], crop=False)  # axs[1:,1])
    make_dff_maps(sucr_main_fly, trials=[1,3,-1], data=map_type, set_title=False,
                  axs=[ax_dict[k] for k in ["I", "L", "O"]], crop=False)  # axs[1:,2])
    """
    ax_dict["X"].axis("off")
    # ax_dict["Y"].axis("off")
    fig.tight_layout()
    return fig

def sup_fig_all_waves():
    # fig, axs = plt.subplots(5,3,figsize=(15,15))
    fig = plt.figure(figsize=(15,15))
    layout = """
    AAAAABBBBBCCCCCC
    DDDDDEEEEEFFFFFF
    GGGGGHHHHHIIIIII
    JJJJJKKKKKLLLLLL
    MMMMMNNNNNOOOOOO
    """
    ax_dict = fig.subplot_mosaic(layout)

    make_detailled_wave_plots(high_caff_main_fly, i_trial=-2, title="auto", crop=[35,50],
                              axs=[ax_dict[k] for k in ["B", "A", "C"]])  # axs=axs[0,:],
    make_detailled_wave_plots(high_caff_main_fly, i_trial=-1, title="auto", crop=[35,50], legend=False,
                              axs=[ax_dict[k] for k in ["E", "D", "F"]])  # axs=axs[1,:],
    make_detailled_wave_plots(high_caff_flies[1], i_trial=-2, title="auto", crop=[5,65], legend=False,
                              axs=[ax_dict[k] for k in ["H", "G", "I"]])  # axs=axs[2,:],
    make_detailled_wave_plots(high_caff_flies[1], i_trial=-1, title="auto", crop=[5,65], legend=False,
                              axs=[ax_dict[k] for k in ["K", "J", "L"]])  # axs=axs[3,:],
    make_detailled_wave_plots(high_caff_flies[2], i_trial=-2, title="auto", crop=[35,60], legend=False,
                              axs=[ax_dict[k] for k in ["N", "M", "O"]])  # axs=axs[4,:],
    fig.tight_layout()
    return fig

def fig_contributions():
    fig, ax = plt.subplots(1,1,figsize=(5,3))
    authors = ["LH", "MK", "JB", "VLR", "CLC",
               "SG", "FA", "MSS", "PR"]
    tasks = ["Conceptualization",  "Methodology",  "Software",  "Validation",  "Formal  analysis",
             "Investigation",  "Data acquisition", "Data curation", "Writing: Original Draft", "Writing: Review & Editing",
             "Visualization", "Resources", "Supervision", "Project Administration", "Funding Acquisition"]
    contribs = np.array([
        [1,1,1,1,1, 1,1,1,1,1, 1,0,0,0,0],  # LH
        [0,1,1,1,1, 0,0,0,1,1, 1,0,0,0,0],  # MK
        [0,1,1,0,1, 0,0,1,0,1, 1,0,0,0,0],  # JB
        [0,1,1,0,1, 0,0,1,0,1, 1,0,0,0,0],  # VLR
        [0,0,0,0,0, 0,1,0,0,1, 0,0,0,0,0],  # CLC

        [0,1,1,0,0, 0,0,1,0,1, 0,0,0,0,0],  # SG
        [0,1,1,0,0, 0,0,0,0,1, 0,0,0,0,0],  # FA
        [1,1,0,0,0, 0,0,0,1,1, 0,1,1,1,1],  # MSS
        [1,1,0,0,0, 0,0,0,1,1, 0,1,1,1,1],  # PR
    ])
    ax.imshow(contribs, cmap=plt.cm.get_cmap("binary"), clim=[0,1])

    ax.set_xticks(np.arange(contribs.shape[1]))
    ax.set_yticks(np.arange(contribs.shape[0]))

    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_yticklabels(authors, fontsize=8)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(contribs.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(contribs.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    return fig

def main():
    datestring = datetime.now().strftime("%Y%m%d_%H%M")
    
    with PdfPages(os.path.join(OUTPUT_PATH, f"_figures_paper_{datestring}.pdf")) as pdf:
        for map_type in ["dff", "green", "green_denoised"]:
            fig = main_figure(map_type)
            pdf.savefig(fig)
    plt.close(fig)
    
    with PdfPages(os.path.join(OUTPUT_PATH, f"_supfigures_paper_{datestring}.pdf")) as pdf:
        fig = sup_fig_all_waves()
        pdf.savefig(fig)
    plt.close(fig)
    """
    with PdfPages(os.path.join(OUTPUT_PATH, f"_contribs_{datestring}.pdf")) as pdf:
        fig = fig_contributions()
        pdf.savefig(fig)
    fig.savefig(os.path.join(OUTPUT_PATH, f"_contribs.png"), dpi=300)
    plt.close(fig)
    """

if __name__ == "__main__":
    main()
