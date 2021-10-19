import os, sys

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
OUT_PATH = os.path.join(MODULE_PATH, "outputs")

import numpy as np
import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from twoppp.plot import plot_sample_pixels
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp import load
from twoppp.utils import get_stack


date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)

fly_dir = fly_dirs[0]
params = PreProcessParams()
preprocess = PreProcessFly(fly_dir, params=params)

pixels = [[174, 97], [91, 172], [171, 280], [122, 443], [185, 575]]
# left bottom, left top, center, top right dim, bottom right
pixels_noise = [[50, 50], [129, 272]]

file_dir = os.path.join(OUT_PATH, "210409_example_pixels_210301_J1xCI9_Fly1_std.pdf")
print("plotting std across trials")
with PdfPages(file_dir) as pdf:
    for processed_dir in preprocess.trial_processed_dirs:
        print(processed_dir)
        raw = get_stack(os.path.join(processed_dir, preprocess.params.green_com_warped))[30:-30, 80:-80, 48:-48]
        denoised = get_stack(os.path.join(processed_dir, preprocess.params.green_denoised))
        std_raw = np.std(raw, axis=0)
        std_denoised = np.std(denoised, axis=0)
        quantile = 0.99
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
        axs[0].imshow(std_raw, clim=[0, np.quantile(std_raw, quantile)])
        axs[0].set_title("std projection of raw data 210301")

        axs[1].imshow(std_denoised, clim=[0, np.quantile(std_denoised, quantile)])
        axs[1].set_title("denoised")

        _ = [[ax.plot(pixel[1], pixel[0], 'r+') for pixel in pixels] for ax in axs.flatten()]
        _ = [[ax.plot(pixel[1], pixel[0], 'k*') for pixel in pixels_noise] for ax in axs.flatten()]

        pdf.savefig(fig)
        plt.close(fig)

file_dir = os.path.join(OUT_PATH, "210409_example_pixels_210301_J1xCI9_Fly1.pdf")
print("plotting pixels across trials")
with PdfPages(file_dir) as pdf:
    for processed_dir in preprocess.trial_processed_dirs:
        print(processed_dir)
        raw = get_stack(os.path.join(processed_dir, preprocess.params.green_com_warped))[30:-30, 80:-80, 48:-48]
        denoised = get_stack(os.path.join(processed_dir, preprocess.params.green_denoised))
        fig = plot_sample_pixels([raw, denoised], pixels=pixels, legends=["raw", "denoised"])
        pdf.savefig(fig)
        plt.close(fig)

file_dir = os.path.join(OUT_PATH, "210409_example_pixels_210301_J1xCI9_Fly1_noise.pdf")
print("plotting noise pixels across trials")
with PdfPages(file_dir) as pdf:
    for processed_dir in preprocess.trial_processed_dirs:
        print(processed_dir)
        raw = get_stack(os.path.join(processed_dir, preprocess.params.green_com_warped))[30:-30, 80:-80, 48:-48]
        denoised = get_stack(os.path.join(processed_dir, preprocess.params.green_denoised))
        fig = plot_sample_pixels([raw, denoised], pixels=pixels_noise, legends=["raw", "denoised"])
        pdf.savefig(fig)
        plt.close(fig)

