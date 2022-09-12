"""
sub-module to plot data in 3D and make videos of 3D data
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

from twoppp.utils import get_stack, save_stack, normalise_quantile
from twoppp.plot import videos

def tmp_avg_3d_stacks(green, red, green_avg=None, red_avg=None):
    save_avgs = False
    if green_avg is not None and red_avg is not None:
        try:
            green_avg_T = get_stack(green_avg)
            red_avg_T = get_stack(red_avg)
        except FileNotFoundError:
            save_avgs = True
            print("Could not find green_avg and red_avg files. Will compute from raw tif.")

    green = get_stack(green)
    red = get_stack(red)
    if green is None:
        red_avg_T = np.mean(red, axis=0)
        green_avg_T = np.zeros_like(red_avg_T)
    else:
        green_avg_T = np.mean(green, axis=0)
        red_avg_T = np.mean(red, axis=0) if red is not None else np.zeros_like(green_avg_T)
    if save_avgs:
        if green is not None:
            save_stack(green_avg, green_avg_T)
        if red is not None:
            save_stack(red_avg, red_avg_T)

    return green_avg_T, red_avg_T

def make_avg_videos_3d(green, red, out_dir, green_avg=None, red_avg=None):
    green_avg_T, red_avg_T = tmp_avg_3d_stacks(green, red, green_avg, red_avg)

    videos.make_video_2p(green_avg_T, out_dir=out_dir, video_name="zstack.mp4",
                         red=red_avg_T, percentiles=(5,99), frames=None, frame_rate=60, trial_dir=None)

    videos.make_video_2p(green_avg_T.T, out_dir=out_dir, video_name="xstack.mp4",
                     red=red_avg_T.T, percentiles=(5,99), frames=None, frame_rate=60, trial_dir=None)

    videos.make_video_2p(np.transpose(green_avg_T, (1,0,2)), out_dir=out_dir, video_name="ystack.mp4",
                     red=np.transpose(red_avg_T, (1,0,2)), percentiles=(5,99), frames=None, frame_rate=60, trial_dir=None)

def plot_projections_3d(green, red, out_dir, green_avg=None, red_avg=None):
    green_avg_T, red_avg_T = tmp_avg_3d_stacks(green, red, green_avg, red_avg)
    if np.sum(green_avg_T) == 0:
        use_green = False
    else:
        use_green = True
    if np.sum(red_avg_T) == 0:
        use_red = False
    else:
        use_red = True

    assert use_red or use_green

    fig, axs = plt.subplots(3,3, figsize=(9.5, 9))

    for i_dim, dim_name in enumerate(["Z", "Y", "X"]):
        for i_m, (method, method_name) in enumerate(zip([np.mean, np.max, np.std], ["mean", "max", "std"])):
            if use_green:
                green_red_dim = normalise_quantile(method(green_avg_T, axis=i_dim), q=(0.05, 0.99), axis=None)
            else:
                red_red_dim = normalise_quantile(method(red_avg_T, axis=i_dim), q=(0.05, 0.99), axis=None)
                green_red_dim = np.zeros_like(red_red_dim)
            if use_red:
                red_red_dim = normalise_quantile(method(red_avg_T, axis=i_dim), q=(0.05, 0.99), axis=None)
            else:
                red_red_dim = np.zeros_like(green_red_dim)
            rgb_im = videos.rgb(red_red_dim, green_red_dim, green_red_dim, None)

            axs[i_dim, i_m].imshow(rgb_im)
            axs[i_dim, i_m].set_title(f"{dim_name} {method_name} projection")
            axs[i_dim, i_m].spines['right'].set_color('none')
            axs[i_dim, i_m].spines['top'].set_color('none')
            axs[i_dim, i_m].spines['left'].set_color('none')
            axs[i_dim, i_m].spines['bottom'].set_color('none')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3dproject.png"), dpi=300)

    