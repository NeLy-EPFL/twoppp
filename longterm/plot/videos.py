# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

import utils2p
import utils_video.generators
from utils_video import make_video
from utils_video.utils import resize_shape, colorbar, add_colorbar

FILE_PATH = os.path.realpath(__file__)
PLOT_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(PLOT_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import get_stack

def make_video_2p(green, out_dir, video_name, red=None, percentiles=(5,99), frames=None, frame_rate=None, trial_dir=None):
    green = get_stack(green)
    if frames is None:
        frames = np.arange(green.shape[0])
    else:
        assert np.sum(frames >= green.shape[0]) == 0
        green = green[frames, :, :]
    if red is None:
        red = np.zeros_like(green[frames, :, :])
    else:
        red = get_stack(red)
        red = red[frames, :, :]
    assert red.shape==green.shape

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
        
    generator = utils_video.generators.frames_2p(red, green, percentiles=percentiles)
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)
    
    
def generator_dff(stack, size=None, font_size=16, vmin=None, vmax=None):
    # copied and modified from utils_video to allow external definition of vmin and vmax
    vmin = np.percentile(stack, 0.5) if vmin is None else vmin
    vmax = np.percentile(stack, 99.5) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.jet

    if size is None:
        image_shape = stack.shape[1:3]
        cbar_shape = (stack.shape[1], max(math.ceil(stack.shape[2] * 0.1), 150))
    else:
        cbar_width = max(math.ceil(size[1] * 0.1), 150)
        image_shape = (size[0], size[1] - cbar_width)
        image_shape = resize_shape(image_shape, stack.shape[1:3])
        cbar_shape = (image_shape[0], cbar_width)

    cbar = colorbar(norm, cmap, cbar_shape, font_size=font_size)

    def frame_generator():
        for frame in stack:
            frame = cmap(norm(frame))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, image_shape[::-1])
            frame = add_colorbar(frame, cbar, "right")
            yield frame

    return frame_generator()

def make_video_dff(dff, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None,
                   vmin=None, vmax=None):
    dff = get_stack(dff)
    if frames is None:
        frames = np.arange(dff.shape[0])
    else:
        assert np.sum(frames >= dff.shape[0]) == 0
        dff = dff[frames, :, :]


    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    generator = generator_dff(dff, vmin=vmin, vmax=vmax)
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)

def make_multiple_video_2p(greens, out_dir, video_name, reds=None, percentiles=(5,99), frames=None, frame_rate=None, trial_dir=None):
    if not isinstance(greens, list):
        greens = [greens]
    greens = [get_stack(green) for green in greens]
    if frames is None:
        frames = np.arange(greens[0].shape[0])
    else:
        assert np.sum(frames >= greens[0].shape[0]) == 0
        greens = [green[frames, :, :] for green in greens]

    if not isinstance(reds, list):
        reds = [reds]
    if reds[0] is None:
        reds = [np.zeros_like(green[frames, :, :]) for green in greens]
    else:
        reds = [get_stack(red) for red in reds]
        reds = [red[frames, :, :] for red in reds]
    assert all([red.shape==green.shape for red, green in zip(reds, greens)])


    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    generators = [utils_video.generators.frames_2p(red, green, percentiles=percentiles) for green, red in zip(greens, reds)]
    generator = utils_video.generators.stack(generators, axis=1)
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)


def make_multiple_video_dff(dffs, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None,
                            vmin=None, vmax=None):
    if not isinstance(dffs, list):
        dffs = [dffs]
    dffs = [get_stack(dff) for dff in dffs]
    if frames is None:
        frames = np.arange(dffs[0].shape[0])
    else:
        assert np.sum(frames >= dffs[0].shape[0]) == 0
        dffs = [dff[frames, :, :] for dff in dffs]    

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    generators = [generator_dff(dff, vmin=vmin, vmax=vmax) for dff in dffs]
    generator = utils_video.generators.stack(generators, axis=1)
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)
