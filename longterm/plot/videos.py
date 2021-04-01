# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from torch import from_numpy

import utils2p
import utils_video.generators
from utils_video import make_video
from utils_video.utils import resize_shape, colorbar, add_colorbar

FILE_PATH = os.path.realpath(__file__)
PLOT_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(PLOT_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import get_stack, torch_to_numpy
from longterm import load
from longterm.utils.color_wheel import flow_to_color
from longterm.register.warping import apply_motion_field, apply_offset
from longterm.register.deepreg.model import Warper


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

    vmins = [np.percentile(dff, 0.5) if vmin is None else vmin for dff in dffs]
    vmaxs = [np.percentile(dff, 99.5) if vmax is None else vmax for dff in dffs]
    vmin = np.min(vmins)
    vmax = np.max(vmax)

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

def generator_motion_field_colorwheeel(motion_fields):
    def frame_generator():
        for field in motion_fields:
            colorwheel = flow_to_color(field)
            yield colorwheel
    return frame_generator()

def generator_motion_field_grid(motion_fields, line_distance=5, warping="dnn"):
    N_frames, N_y, N_x, _ = motion_fields.shape
    grids = np.zeros((N_frames, N_y, N_x))
    grids[:, np.arange(0, N_y, line_distance), :] = 1
    grids[:, :, np.arange(0, N_x, line_distance)] = 1

    if warping == "dnn":
        grids = from_numpy(grids[:, np.newaxis, :, :]).float().cuda()
        motion_fields = from_numpy(np.moveaxis(motion_fields, -1, 1)).float().cuda()
        warper = Warper()
        grids_applied = warper(grids, motion_fields)
        grids_applied = np.squeeze(torch_to_numpy(grids_applied))
    elif warping == "ofco":
        grids_applied = apply_motion_field(grids, motion_fields)

    cmap = plt.cm.binary

    def frame_generator():
        for frame in grids_applied:
            frame = cmap(frame)
            frame = (frame * 255).astype(np.uint8)
            yield frame
    return frame_generator()

def make_video_motion_field(motion_fields, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None, 
                            visualisation="grid", line_distance=5, warping="dnn"):  # other options: "colorwheel"
    motion_fields = get_stack(motion_fields)
    if frames is None:
        frames = np.arange(motion_fields.shape[0])
    else:
        assert np.sum(frames >= motion_fields.shape[0]) == 0
        motion_fields = motion_fields[frames, :, :, :]

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
    if visualisation == "colorwheel":
        generator = generator_motion_field_colorwheeel(motion_fields)
    elif visualisation == "grid":
        generator = generator_motion_field_grid(motion_fields, line_distance, warping)
    else:
        raise NotImplementedError
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)

def make_multiple_video_motion_field(motion_fields, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None,
                                     visualisation="grid", line_distance=5, warping="dnn"):
    if not isinstance(motion_fields, list):
        motion_fields = [motion_fields]
    motion_fields = [get_stack(motion_field) for motion_field in motion_fields]
    if frames is None:
        frames = np.arange(motion_fields[0].shape[0])
    else:
        assert np.sum(frames >= motion_fields[0].shape[0]) == 0
        motion_fields = [motion_field[frames, :, :] for motion_field in motion_fields]    

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    if visualisation == "colorwheel":
        generators = [generator_motion_field_colorwheeel(motion_field) for motion_field in motion_fields]
    elif visualisation == "grid":
        generators = [generator_motion_field_grid(motion_field, line_distance, warping) for motion_field in motion_fields]
    else:
        raise NotImplementedError
    generator = utils_video.generators.stack(generators, axis=1)
    utils_video.make_video(os.path.join(out_dir, video_name), generator, frame_rate)

if __name__ == "__main__":

    JB_DATA = False
    LH_DATA = False
    DENOISED = False
    PIPELINE = True
    if JB_DATA:
        date_dir = os.path.join(load.LOCAL_DATA_DIR, "210301_J1xCI9")  # 210216_J1xCI9 fly1 trial 0
        fly_dirs = load.get_flies_from_datedir(date_dir)
        trial_dirs = load.get_trials_from_fly(fly_dirs)
        i_trial = 0

        green = os.path.join(trial_dirs[0][i_trial], load.PROCESSED_FOLDER, load.RAW_GREEN_TIFF)
        red = os.path.join(trial_dirs[0][i_trial], load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
        out_dir = os.path.join(trial_dirs[0][i_trial], load.PROCESSED_FOLDER)
        video_name = "raw.mp4"
        make_video_2p(green, out_dir, video_name, red=red, percentiles=(5,99), 
                        frames=np.arange(1000), frame_rate=None, trial_dir=trial_dirs[0][0])

    elif LH_DATA:
        date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
        fly_dirs = load.get_flies_from_datedir(date_dir)
        trial_dirs = load.get_trials_from_fly(fly_dirs)

        green = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, load.RAW_GREEN_TIFF)
        red = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
        green_warped = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, "green_warped.tif")
        red_warped = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, "red_warped.tif")
        green_warped_com = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, "green_com_warped.tif")
        red_warped_com = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, "red_com_warped.tif")
        out_dir = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER)
        """
        make_video_2p(green, out_dir, "raw.mp4", red=red, percentiles=(5,99), 
                        frames=np.arange(500), frame_rate=None, trial_dir=trial_dirs[0][0])
        make_video_2p(green_warped, out_dir, "warped.mp4", red=red_warped, percentiles=(5,99), 
                        frames=np.arange(500), frame_rate=None, trial_dir=trial_dirs[0][0])
        """
        offset_dir = os.path.join(trial_dirs[0][0], load.PROCESSED_FOLDER, "com_offset.npy")
        offsets = np.load(offset_dir)
        green = get_stack(green)
        red = get_stack(red)
        green_centered = apply_offset(green, offsets)
        red_centered = apply_offset(red, offsets)

        make_video_2p(green_centered, out_dir, "com.mp4", red=red_centered, percentiles=(5,99), 
                        frames=np.arange(500), frame_rate=None, trial_dir=trial_dirs[0][0])

        make_video_2p(green_warped_com, out_dir, "com_warped.mp4", red=red_warped_com, percentiles=(5,99), 
                        frames=np.arange(500), frame_rate=None, trial_dir=trial_dirs[0][0])

    elif DENOISED:
        base_dir = "/home/jbraun/bin/deepinterpolation/sample_data"
        greens = [get_stack(os.path.join(base_dir, "longterm_003_crop.tif"))[30:-30],
                  get_stack(os.path.join(base_dir, "denoised_longterm_003_crop_out.tif")),
                  get_stack(os.path.join(base_dir, "denoised_halves_longterm_003_crop_out.tif"))]
        make_multiple_video_2p(greens=greens, out_dir=base_dir, video_name="compare_denoise", frame_rate=8, percentiles=(0, 99))

    elif PIPELINE:
        base_dir = "/home/jbraun/bin/deepinterpolation/sample_data"
        date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
        fly_dirs = load.get_flies_from_datedir(date_dir)
        trial_dirs = load.get_trials_from_fly(fly_dirs)
        trial_dir = trial_dirs[0][0]
        """
        greens = [get_stack(os.path.join(trial_dir, load.PROCESSED_FOLDER, load.RAW_GREEN_TIFF))[30:-30, :, :],
                  get_stack(os.path.join(trial_dir, load.PROCESSED_FOLDER, "green_com_warped.tif"))[30:-30, :, :],
                  get_stack(os.path.join(base_dir, "denoised_randomlrc1_01_longterm_003_crop_out.tif"))]
        reds = [get_stack(os.path.join(trial_dir, load.PROCESSED_FOLDER, load.RAW_RED_TIFF))[30:-30, :, :],
                get_stack(os.path.join(trial_dir, load.PROCESSED_FOLDER, "red_com_warped.tif"))[30:-30, :, :],
                np.zeros_like(greens[2])]

        make_multiple_video_2p(greens=greens, reds=reds, out_dir=os.path.join(trial_dir, load.PROCESSED_FOLDER),
                               video_name="pipeline.mp4", frames=np.arange(8*60), trial_dir=trial_dir)
        """
        make_video_2p(green=os.path.join(trial_dir, load.PROCESSED_FOLDER, "dn640_warped_green.tif"),
                      red=os.path.join(trial_dir, load.PROCESSED_FOLDER, "dn640_warped_red.tif"),
                      video_name="dn640_warped.mp4", out_dir=os.path.join(trial_dir, load.PROCESSED_FOLDER),
                      trial_dir=trial_dir, frames=np.arange(500))
        pass