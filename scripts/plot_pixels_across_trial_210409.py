import os, sys

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
OUT_PATH = os.path.join(MODULE_PATH, "outputs")

import matplotlib
matplotlib.use('agg')  # use non-interactive backend for PNG plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from longterm.plot import plot_sample_pixels
from longterm.pipeline import PreProcessFly, PreProcessParams
from longterm import load
from longterm.utils import get_stack


date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)

fly_dir = fly_dirs[0]
params = PreProcessParams()
preprocess = PreProcessFly(fly_dir, params=params)

pixels = [[174, 97], [91, 172], [171, 280], [122, 443], [185, 575]]
# left bottom, left top, center, top right dim, bottom right
pixels_noise = [[50, 50], [129, 272]]

file_dir = os.path.join(OUT_PATH, "210409_example_pixels_210301_J1xCI9_Fly1.pdf")
with PdfPages(file_dir) as pdf:
    for processed_dir in preprocess.trial_processed_dirs:
        raw = get_stack(os.path.join(processed_dir, preprocess.params.green_com_warped))[30:-30, :, :]
        denoised = get_stack(os.path.join(processed_dir, preprocess.params.green_denoised))
        fig = plot_sample_pixels([raw, denoised], pixels=pixels, legends=["raw", "denoised"])
        pdf.savefig(fig)
        plt.close(fig)

file_dir = os.path.join(OUT_PATH, "210409_example_pixels_210301_J1xCI9_Fly1_noise.pdf")
with PdfPages(file_dir) as pdf:
    for processed_dir in preprocess.trial_processed_dirs:
        raw = get_stack(os.path.join(processed_dir, preprocess.params.green_com_warped))[30:-30, :, :]
        denoised = get_stack(os.path.join(processed_dir, preprocess.params.green_denoised))
        fig = plot_sample_pixels([raw, denoised], pixels=pixels_noise, legends=["raw", "denoised"])
        pdf.savefig(fig)
        plt.close(fig)