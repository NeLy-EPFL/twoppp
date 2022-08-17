"""
Sub-module including generators for videos and functions to make videos.
Video generation is based on utils_video:
https://github.com/NeLy-EPFL/utils_video
"""

import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import cv2
import json
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import pickle
from glob import glob
# from torch import from_numpy

import utils2p
import utils2p.synchronization
import utils_video.generators
# from utils_video import make_video
from utils_video.utils import resize_shape, colorbar, add_colorbar, rgb, process_2p_rgb, get_generator_shape
from deepfly.CameraNetwork import CameraNetwork

from twoppp.utils import get_stack, find_file, crop_img, crop_stack
from twoppp import load
from twoppp.register.warping import apply_motion_field, apply_offset

def make_video(video_path, frame_generator, fps, output_shape=(-1, 1920), n_frames=-1, use_handbrake=True):
    """Wrapper around utils_video.make_video() changing the frame rate to be a non-integer value
    to avoid warning and potential error.
    """
    print(f"Making video with {n_frames} frames.")
    if np.log2(fps) % 1 == 0:
        fps += 0.01
    utils_video.make_video(video_path, frame_generator, fps,
                           output_shape=output_shape, n_frames=n_frames)

    if use_handbrake:
        handbrake(video_path)


def handbrake(video_path, output_path=None):
    """apply HandBrake to a video to compress it.
    This is an EXPERIMENTAL FEATURE and requires that the Handbrake command line
    interface is installed!
    For installation follow the following steps:
    Download the command line interface (CLI) from here:
    https://handbrake.fr/downloads2.php
    using the following instructions:
    https://handbrake.fr/docs/en/1.5.0/get-handbrake/download-and-install.html
    >>> flatpak --user install HandBrakeCLI-1.4.2-x86_64.flatpak
    You might have to install flatpak with apt-get first.
    If error about unacceptable TLS certificate pops up:
    >>> sudo apt install --reinstall ca-certificates
    Add flatpak to your PATH. This way you can use the commands below
    >>> export PATH=$PATH:$HOME/.local/share/flatpak/exports/bin:/var/lib/flatpak/exports/bin
    Now the CLI can be run as follows:
    https://handbrake.fr/docs/en/latest/cli/cli-options.html
    >>> fr.handbrake.HandBrakeCLI -i source -o destination
    Parameters
    ----------
    video_path : str
        path to your .mp4 video file
    output_path : str, optional
        where the compressed video should be saved to. If None, overwrite original video, by default None
    """
    if output_path is None:
        REPLACE = True
        folder, file_name = os.path.split(video_path)
        output_path = os.path.join(folder, "tmp_" + file_name)
    else:
        REPLACE = False
    
    export_path = "export PATH=$PATH:$HOME/.local/share/flatpak/exports/bin:/var/lib/flatpak/exports/bin"
    # check whether handbrake CLI is installed
    if os.system(export_path+" && fr.handbrake.HandBrakeCLI -h >/dev/null 2>&1"):
        print("HandBrakeCLI is not installed.\n",
              "Install files can be found here: https://handbrake.fr/downloads2.php \n",
              "Install instructions here: https://handbrake.fr/docs/en/1.5.0/get-handbrake/download-and-install.html")
        return
    # run the client on the video
    os.system(export_path+f" && fr.handbrake.HandBrakeCLI -i {video_path} -o {output_path}")
    if REPLACE:
        os.system(f"mv {output_path} {video_path}")

def make_video_2p(green, out_dir, video_name, red=None, percentiles=(5,99),
                  frames=None, frame_rate=None, trial_dir=None):
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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
        
    generator = utils_video.generators.frames_2p(red, green, percentiles=percentiles)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)
    
def generator_frames_2p(red_stack, green_stack, percentiles=(5, 95), red_vlim=None, green_vlim=None):
    channels = []
    v_max = []
    v_min = []
    if isinstance(percentiles[0], list) or isinstance(percentiles[0], tuple):
        red_percentiles = percentiles[0]
        green_percentiles = percentiles[1]
    else:
        red_percentiles = percentiles
        green_percentiles = percentiles

    if red_stack is not None:
        channels.append("r")
        if red_vlim is None:
            v_max.append(np.percentile(red_stack, red_percentiles[1]))
            v_min.append(np.percentile(red_stack, red_percentiles[0]))
        else:
            v_max.append(red_vlim[1])
            v_min.append(red_vlim[0])
        ASSIGN_RED_STACK = False
    else:
        ASSIGN_RED_STACK = True
    if green_stack is not None:
        channels.append("g")
        if green_vlim is None:
            v_max.append(np.percentile(green_stack, green_percentiles[1]))
            v_min.append(np.percentile(green_stack, green_percentiles[0]))
        else:
            v_max.append(green_vlim[1])
            v_min.append(green_vlim[0])
        channels.append("b")
        v_max.append(v_max[-1])
        v_min.append(v_min[-1])
    else:
        green_stack = [None for i in range(len(red_stack))]

    if ASSIGN_RED_STACK:
        red_stack = [None for i in range(len(green_stack))]

    for red_frame, green_frame in zip(red_stack, green_stack):
        frame = rgb(red_frame, green_frame, green_frame, None)
        frame = process_2p_rgb(frame, channels, v_max, v_min)
        frame = frame.astype(np.uint8)
        yield frame

def generator_dff(stack, size=None, font_size=16, pmin=0.5, pmax=99.5, vmin=None, vmax=None,
                  blur=0, mask=None, crop=None, log_lim=False,
                  text=None, colorbarlabel="dff"):
    """generator to make dff videos.
    Modified from https://github.com/NeLy-EPFL/utils_video

    Parameters
    ----------
    stack : numpy array or str
        stack of dff frames or path pointing to .ti

    size : tuple, optional
        resize video to given size, by default None

    font_size : int, optional
        text font size, by default 16

    pmin : float, optional
        percentage min of dff, overruled by vmin by default 0.5

    pmax : float, optional
        percentage max of dff, overruled by vmax by default 99.5

    vmin : float, optional
        absolute minimum values for dff, if not specified, use pmin, by default None

    vmax : float, optional
        absolute maximum values for dff. if not specified, use pmax, by default None

    blur : float, optional
        whether to spatially blur dff. width of Gaussian kernel, by default 0

    mask : numpy array or str, optional
        mask to apply over the dff, by default None

    crop : list, optional
        cropping applied to dff
        list of length 2 for symmetric cropping (same on both sides),
        or list of length 4 for assymetric cropping, by default None

    log_lim : bool, optional
        EXPERIMENTAL FUNCTION: whether to use logarithmic limits and scale for dff,
        by default False

    text : str, optional
        text to show in top left of image, by default None

    colorbarlabel : str, optional
        text printed next to the colourbar, by default "dff"

    Yields
    -------
    frame: numpy array
    """
    colorbarlabel = r"%$\frac{\Delta F}{F}$" if colorbarlabel == "dff" else colorbarlabel
    if mask is not None:
        mask = np.invert(mask)  # invert only once
        stack_tmp = np.zeros_like(stack)
        for i_f, f in enumerate(stack):
            f[mask] = 0
            stack_tmp[i_f, :, :] = f
        stack = stack_tmp
    stack = crop_stack(stack, crop)

    vmin = np.percentile(stack, pmin) if vmin is None else vmin
    vmax = np.percentile(stack, pmax)if vmax is None else vmax
    if log_lim:
        norm = colors.LogNorm(vmin=np.maximum(0.1,vmin), vmax=vmax)
    else:
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
    try:
        cbar = colorbar(norm, cmap, cbar_shape, font_size=font_size, label=colorbarlabel)
    except:
        print("Using old version of utils_video. please update utils_video")
        cbar = colorbar(norm, cmap, cbar_shape, font_size=font_size)

    def frame_generator():
        for frame in stack:
            if blur:
                frame = gaussian_filter(frame, (blur, blur))
            frame = cmap(norm(frame))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, image_shape[::-1])
            frame = add_colorbar(frame, cbar, "right")
            yield frame

    generator = frame_generator()
    if text is not None:
        generator = utils_video.generators.add_text(generator, text=text, pos=(10,50))
    return generator

def make_video_dff(dff, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None,
                   vmin=0, vmax=None, pmin=1, pmax=99, blur=0, mask=None, crop=None, text=None):
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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    generator = generator_dff(dff, vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, 
                                     blur=blur, mask=mask, crop=crop, text=text)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)

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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    N_frames = np.max([len(green) for green in greens])
    generators = [utils_video.generators.frames_2p(red, green, percentiles=percentiles) for green, red in zip(greens, reds)]
    generator = utils_video.generators.stack(generators, axis=1, allow_different_length=True)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)

def make_multiple_video_dff(dffs, out_dir, video_name, frames=None, frame_rate=None, trial_dir=None,
                            vmin=0, vmax=None, pmin=1, pmax=99, share_lim=True, blur=0, crop=None, 
                            mask=None, share_mask=False, text=None):
    if not isinstance(dffs, list):
        dffs = [dffs]
    if not isinstance(dffs[0], list):
        dffs = [dffs]
    if text is not None:
        if not isinstance(text, list):
            text = [text]
        if not isinstance(text[0], list):
            text = [text]
        assert len(text) == len(dffs)
        assert len(text[0]) == len(text[0])
    else:
        text = [[None for _ in row] for row in dffs]
    if share_mask or mask is None:
        mask = [[mask for _ in row] for row in dffs]
    assert len(mask) == len(dffs)
    assert len(mask[0]) == len(text[0])
    
    dffs = [[get_stack(dff) for dff in dffs_] for dffs_ in dffs]
    if frames is None:
        frames = np.arange(dffs[0][0].shape[0])
    else:
        assert np.sum(frames >= dffs[0][0].shape[0]) == 0
        dffs = [[dff[frames, :, :] for dff in dffs_] for dffs_ in dffs]
    dffs_tmp = []
    for mask_row, dff_row in zip(mask, dffs):
        dff_row_tmp = []
        for this_mask, this_dff in zip(mask_row, dff_row):
            if this_mask is not None:
                this_mask = np.invert(this_mask)  # invert only once
                this_dff_tmp = np.zeros_like(this_dff)
                for i_f, f in enumerate(this_dff):
                    f[this_mask] = 0
                    this_dff_tmp[i_f, :, :] = f
                dff_row_tmp.append(this_dff_tmp)
            else:
                dff_row_tmp.append(this_dff)
        dffs_tmp.append(dff_row_tmp)
    dffs = dffs_tmp
    

    if share_lim:
        vmins = [[np.percentile(dff, pmin) if vmin is None else vmin for dff in dffs_] for dffs_ in dffs]
        vmaxs = [[np.percentile(dff, pmax) if vmax is None else vmax for dff in dffs_] for dffs_ in dffs]
        vmin = np.mean(vmins)
        vmax = np.mean(vmaxs)

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
    N_frames = np.max([[len(dff) for dff in dff_] for dff_ in dffs])
    generators = [[generator_dff(dff, vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax,
                                 blur=blur, crop=crop, mask=None, text=t)
                   for dff, t in zip(dffs_, text_)]
                  for dffs_, text_ in zip(dffs, text)]
    generator_rows = [utils_video.generators.stack(generator_row, axis=1, allow_different_length=True)
                      for generator_row in generators]
    generator = utils_video.generators.stack(generator_rows, axis=0, allow_different_length=True)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)

def generator_motion_field_grid(motion_fields, line_distance=5, warping="dnn"):
    N_frames, N_y, N_x, _ = motion_fields.shape
    grids = np.zeros((N_frames, N_y, N_x))
    grids[:, np.arange(0, N_y, line_distance), :] = 1
    grids[:, :, np.arange(0, N_x, line_distance)] = 1

    if warping == "dnn":
        raise NotImplementedError("Currently only 'ofco' is implemented.")
        # grids = from_numpy(grids[:, np.newaxis, :, :]).float().cuda()
        # motion_fields = from_numpy(np.moveaxis(motion_fields, -1, 1)).float().cuda()
        # warper = Warper()
        # grids_applied = warper(grids, motion_fields)
        # grids_applied = np.squeeze(torch_to_numpy(grids_applied))
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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
    if visualisation == "colorwheel":
        raise NotImplementedError("Currently only 'grid' is implemented.")
    elif visualisation == "grid":
        generator = generator_motion_field_grid(motion_fields, line_distance, warping)
    else:
        raise NotImplementedError("Currently only 'grid' is implemented.")
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)

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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    if visualisation == "colorwheel":
        raise NotImplementedError("Currently only 'grid' is implemented.")
    elif visualisation == "grid":
        generators = [generator_motion_field_grid(motion_field, line_distance, warping) for motion_field in motion_fields]
    else:
        raise NotImplementedError("Currently only 'grid' is implemented.")
    generator = utils_video.generators.stack(generators, axis=1)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)

def generator_df3d(image_folder, cameras=[5,3,1], font_size=16, print_frame_num=True, print_frame_time=True, 
                   print_beh_label=False, beh_label_dir=None,
                   N_frames=None, frame_rate=None, factor_downsample=1, speedup=None):
    N_frames = -1 if N_frames is None else N_frames
    camNet = CameraNetwork(image_folder=image_folder, output_folder=os.path.join(image_folder, 'df3d'), num_images=N_frames)
    N_frames = camNet.cam_list[0].points2d.shape[0] if N_frames == -1 else N_frames
    camNet.num_images = N_frames

    cmap = plt.cm.get_cmap("seismic")
    i_cs = np.linspace(start=0, stop=1, num=10)
    i_cs[5:] = np.flip(i_cs[5:])
    colors = [np.array(cmap(i_c))*255 for i_c in i_cs]
    if os.path.isfile(os.path.join(image_folder, f"camera_{cameras[0]}_img_0.jpg")):
        saved_as_frames = True
        # CameraNetwork can read frames already, but if no frames are saves, then need to load videos
    else:
        saved_as_frames = False
        caps = [cv2.VideoCapture(os.path.join(image_folder, f"camera_{cam}.mp4")) for cam in cameras]
        videos = [[] for _ in cameras]
        for i_cam, (cam, cap) in enumerate(zip(cameras, caps)):
            i_frame = 0
            while(1):
                ret, frame = cap.read()
                videos[i_cam].append(frame)
                i_frame += 1
                if cv2.waitKey(1) & 0xFF == ord('q') or ret==False or i_frame >= N_frames:
                    cap.release()
                    break

    def single_cam_frame_generator(cam):
        i_cam = np.argwhere(np.array(cameras)==cam)[0,0]
        cam_df3d = np.argwhere(np.array(camNet.cid2cidread)==cam)[0,0]
        for i_frame in range(0, N_frames, factor_downsample):
            img = videos[i_cam][i_frame] if not saved_as_frames else None
            frame = camNet.cam_list[cam_df3d].plot_2d(img_id=i_frame, colors=colors, img=img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    if len(cameras) == 1:
        generator = single_cam_frame_generator(cameras[0])
    else:
        generators = [single_cam_frame_generator(cam) for cam in cameras]
        generator = utils_video.generators.stack(generators, axis=1)

    if print_frame_num:
        text = ["frame: {:5d}".format(i) for i in range(0, N_frames, factor_downsample)]
        generator = utils_video.generators.add_text(generator, text=text, pos=(10,100))
    if print_frame_time:
        if frame_rate is None:
            metadata_dir = utils2p.find_seven_camera_metadata_file(image_folder)
            with open(metadata_dir, "r") as f:
                metadata = json.load(f)
            frame_rate = metadata["FPS"]
            del metadata
        text = ["time: {:5.1f} s".format(i/frame_rate) for i in range(0, N_frames, factor_downsample)]
        generator = utils_video.generators.add_text(generator, text=text, pos=(10,150))
    if print_beh_label:
        beh_labels = pd.read_pickle(beh_label_dir)
        assert len(beh_labels) >= N_frames
        text = beh_labels["Prediction"][0:N_frames:factor_downsample].values
        generator = utils_video.generators.add_text(generator, text=text, pos=(10,200))
    if speedup is not None:
        text = "{}x".format(speedup)
        generator = utils_video.generators.add_text(generator, text=text, pos=(10,50))
    return generator

def make_video_df3d(trial_dir, out_dir, video_name, frames=None, frame_rate=None, cameras=[5,1], 
                    print_frame_num=True, print_frame_time=True, print_beh_label=False, beh_label_dir=None,
                    downsample=None, speedup=1):
    images_dir = os.path.join(trial_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(trial_dir, "behData", "images")
        if not os.path.isdir(images_dir):
            images_dir = find_file(trial_dir, "images", "images folder")
            if not os.path.isdir(images_dir):
                raise FileNotFoundError("Could not find 'images' folder.")
    if frame_rate is None:
        metadata_dir = utils2p.find_seven_camera_metadata_file(trial_dir)
        with open(metadata_dir, "r") as f:
            metadata = json.load(f)
        frame_rate = metadata["FPS"]
        del metadata

    factor_downsample = 1 if downsample is None else downsample
    
    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    if not isinstance(frames, int) and len(frames):
        if frames[0] != 0 or any(np.diff(frames) != 1):
            raise NotImplementedError("frames has to be a range of frames starting from 0 or an integer frame length.")
        else:
            frames = len(frames)

    generator = generator_df3d(images_dir, cameras=cameras, frame_rate=frame_rate, N_frames=frames,
                               print_frame_time=print_frame_time, print_frame_num=print_frame_num,
                               print_beh_label=print_beh_label, beh_label_dir=beh_label_dir,
                               factor_downsample=factor_downsample, speedup=speedup)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate/factor_downsample*speedup)

def generator_video(path, size=None, start=0, stop=9223372036854775807, try_frames=True,
                    required_n_frames=None):
    """load video from file and return as generator

    Parameters
    ----------
    path : str
        path of video file

    size : tuple, optional
        if specified, resize image to given size, by default None

    start : int, optional
        which frame to start at, by default 0

    stop : int, optional
        which frame to stop at, by default 9223372036854775807

    try_frames: bool, optional
        whether to try finding individual frames in case the video file was not found

    required_n_frames: int, optional
        number of camera frames that have to be found, by default None

    Yields
    -------
    frame: numpy array

    Raises
    ------
    RuntimeError
        if video is already opened somwhere else
    """
    if os.path.isfile(path):
        try:
            cap = cv2.VideoCapture(path)

            if cap.isOpened() == False:
                raise RuntimeError(f"Error opening video stream or file at {path}.")

            current_frame = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True and current_frame >= start and current_frame < stop:
                    if size is not None:
                        shape = resize_shape(size, frame.shape[:2])
                        frame = cv2.resize(frame, shape[::-1])
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    yield frame
                elif ret == False or current_frame >= stop:
                    break
                current_frame += 1
        finally:
            cap.release()
    elif try_frames:
        print("Video file not found. Will try to find .jpg frames instead.")
        try:
            images_dir, file_name = os.path.split(path)
            camera = int(file_name[-5])  # assume file is named 'camera_5.mp4' for example
            image_list_sorted = find_cam_frames(images_dir, camera, required_n_frames)
            beh_generator = generator_cam_frames(image_list_sorted, start=start,
                                                 stop=stop, size=size)
            for frame in beh_generator:
                yield frame
        except:
            raise FileNotFoundError(f"Could neither find video {path} nor .jpg frames in same path")
    else:
        raise FileNotFoundError(f"Could not find video: {path}")

def find_cam_frames(images_dir, camera, required_n_frames=None):
    """find and sort camera frames according to format string
    f"camera_{camera}_img_*.jpg"
    This is the default output of the 7 cam recording setup.

    Parameters
    ----------
    images_dir : str
        directory to be searched in

    camera : int
        number of camera for which the frames are to be found

    required_n_frames: int, optional
        number of camera frames that have to be found, by default None

    Returns
    -------
    image_list_sorted: list of str
        list containing the sorted paths for each camera frame

    Raises
    ------
    AssertionError
        if a frame is missing
        or if the number of frames found != required_n_frames
    """
    image_list = glob(os.path.join(images_dir, f"camera_{camera}_img_*.jpg"))
    image_numbers = np.array([int(os.path.basename(image)[13:-4]) for image in image_list])
    sort_inds = np.argsort(image_numbers)
    sorted_numbers = image_numbers[sort_inds]
    if required_n_frames is not None and isinstance(required_n_frames, int):
        assert len(image_list) == required_n_frames
    # check that frame numbers are monotonically increasing and no frame is missing
    assert all(np.diff(sorted_numbers) == 1)
    print(f"Found {len(image_list)} camera frames with" + \
          f"frame numbers {sorted_numbers[0]} to {sorted_numbers[-1]}")
    image_list_sorted = [image_list[i_im] for i_im in sort_inds]
    return image_list_sorted

def generator_cam_frames(frames, size=None, start=0, stop=9223372036854775807):
    """read frames from list of images and return as generator

    Parameters
    ----------
    frames : list
        list of frame names

    size : tuple, optional
        if specified, resize image to given size, by default None

    start : int, optional
        which frame to start at, by default 0

    stop : int, optional
        which frame to stop at, by default 9223372036854775807

    Yields
    -------
    frame: numpy array
    """
    for current_frame, frame in enumerate(frames):
        if current_frame >= start and current_frame < stop:
            frame = cv2.imread(frame)
            if size is not None:
                shape = resize_shape(size, frame.shape[:2])
                frame = cv2.resize(frame, shape[::-1])
            yield frame
        elif current_frame >= stop:
            break

def make_video_from_cam_frames(images_dir, camera, required_n_frames=None, video_name="camera"):
    """make a video out of a series of frames acquired by one of the cameras of the 7 camera system.
    Will save in the same folder as where the images are searched

    Parameters
    ----------
    images_dir : str
        folder where the images are to be found

    camera : int
        number of the camera

    required_n_frames : int, optional
        if known, procide the expected number of frames. The function will check that all were found.
        by default None

    video_name : str, optional
        name base of the output video. Will append "_{camera}.mp4", by default "camera"
    """
    cam_frames = find_cam_frames(images_dir, camera, required_n_frames)
    generator = generator_cam_frames(cam_frames)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(images_dir)
    with open(seven_camera_metadata_file, "r") as f:
        metadata = json.load(f)
    fps = metadata["FPS"]
    frame_size = (metadata["ROI"]["Height"][f"{camera}"], metadata["ROI"]["Width"][f"{camera}"])
    utils_video.make_video(os.path.join(images_dir, video_name+f"_{camera}.mp4"), generator,
                           fps=fps, output_shape=frame_size, n_frames=-1)

def downsample_generator(generator, factor):
    """return every Nth element of a genertor

    Parameters
    ----------
    generator : generator
    
    factor : int
        every Nth frame will be returned

    Yields
    -------
    frame: numpy array
    """
    i_frame = 0
    while 1:
        i_frame += 1
        frame = next(generator)
        if i_frame < factor:
            continue
        else:
            i_frame = 0
            yield frame

def selected_frames_generator(generator, select_frames):
    """return generator that only shows selected frames.
    Can also be used to repeat a short sequence over and over again.
    If an error occurs, will continue to yield the first frame.

    Parameters
    ----------
    generator : generator
    
    select_frames : list or numpy array
        indices of frames to show in which sequence. 
        individual indices can appear multiple times

    Yields
    -------
    frame: numpy array
    """
    i_frame = 0
    except_frame = None
    frames = []
    saved_i_frames = []
    first_except = True
    unique_frames, counts = np.unique(select_frames, return_counts=True)
    def frame_reused(i_frame):
        i_unique = np.where(unique_frames==i_frame)[0][0]
        count = counts[i_unique]
        return count > 1
    i_select_frame = 0
    in_repeat = False
    i_repeat = 0
    while 1:
        try:
            if not in_repeat:
                # first cycle through the frames of the generator
                frame = next(generator)
                if i_frame == 0:
                    except_frame = np.zeros_like(frame)
                if i_frame != select_frames[i_select_frame] and not i_frame > unique_frames[-1]:
                    # the current frame is not the next frame
                    if i_frame in select_frames:
                        # frame is not the right frame right now, but might be used later
                        saved_i_frames.append(i_frame)
                        frames.append(frame)
                    i_frame += 1
                    continue
                elif i_frame > unique_frames[-1]:
                    # end of first cycle. switch to looking at stored frames
                    in_repeat = True
                    saved_i_frames = np.array(saved_i_frames)
                    continue
                else:
                    # this is the right frame
                    # store it for later in case it's needed
                    # yield it
                    if frame_reused(i_frame):
                        saved_i_frames.append(i_frame)
                        frames.append(frame)
                    i_frame += 1
                    i_select_frame += 1
                    yield frame
            elif in_repeat:
                store_ind = np.where(saved_i_frames==select_frames[i_select_frame])[0][0]
                frame = frames[store_ind]
                i_select_frame += 1
                yield frame
        except:
            # yield except_frame
            if first_except:
                first_except = False
                frames = [except_frame] + frames
                i_frame = 0
                N_frames = len(frames)
            elif i_frame >= N_frames:
                i_frame = 0
            frame = frames[i_frame]
            i_frame += 1
            yield frame

def make_video_raw_dff_beh(dff, trial_dir, out_dir, video_name, beh_dir=None, sync_dir=None,
                           camera=6, stack_axis=0, green=None, red=None, colorbarlabel="dff",
                           vmin=0, vmax=None, pmin=1, pmax=99, blur=0, mask=None, crop=None,
                           log_lim=False, text=None, text_loc="dff", asgenerator=False,
                           downsample=None, select_frames=None, max_length=None, time=True,
                           stim_start=None, stim_stop=None, twop_percentiles=(5,99)):
    """make a video containing behavioural data and deltaF/F and/or green+red frames.
    Synchronises two photon and behavioural recordings.
    Can be used as a generator, for example when stacking videos of multiple trials

    Parameters
    ----------
    dff : numpy array or str
        delta F/F either as numpy array or as absolute path pointing to a .tif

    trial_dir : str
        directory containing the 2p data and ThorImage output

    out_dir : str
        directory to save the video in

    video_name : str
        name of the output video

    beh_dir : str, optional
         directory containing the 7 camera data. If not specified, will be set equal
        to trial_dir, by default None

    sync_dir : str, optional
        directory containing the output of ThorSync. If not specified, will be set equal
        to trial_dir, by default None

    camera : int, optional
        which camera to use. will search for "camera_X.mp4", by default 6

    stack_axis : int, optional
        along which axis to stack the generators: 0=y axis, 1=x axis, by default 0

    green : numpy array or str, optional
        green stack or path to ,tif, by default None

    red : numpy array or str, optional
        red stack or path to .tif, by default None

    colorbarlabel : str, optional
        name of the dff colourbar label, by default "dff"

    vmin : float, optional
        absolute minimum values for dff, if not specified, use pmin, by default 0

    vmax : float, optional
        absolute maximum values for dff. if not specified, use pmax, by default None

    pmin : float, optional
        percentage min of dff, overruled by vmin by default 1

    pmax : float, optional
        percentage max of dff, overruled by vmax by default 99

    blur : float, optional
        whether to spatially blur dff. width of Gaussian kernel, by default 0

    mask : numpy array or str, optional
        mask to apply over the dff, by default None

    crop : list, optional
        cropping applied to dff
        list of length 2 for symmetric cropping (same on both sides),
        or list of length 4 for assymetric cropping, by default None

    log_lim : bool, optional
        EXPERIMENTAL FUNCTION! whether to use logarithmic limits and scale for dff, by default False

    text : str, optional
        string to write on top of the video, by default None

    text_loc : str, optional
        where the text will be displayed. either "dff" or "beh".
        If "beh", make sure to select time=False, by default "dff"

    asgenerator : bool, optional
        if True, will return video as generator. if not, will make video using make_video(),
        by default False

    downsample : int, optional
        downsampling factor, by default None

    select_frames : list or numpy array, optional
        list containing a sequence of selected two-photon frame inidices.
        if an empty list, generate a black frame. If None, use all frames, by default None

    max_length : int, optional
        maximum length of the video in frames, by default None

    time : bool, optional
        whether or not to show the current time of the recording in the behaviour video,
        by default True

    twop_percentiles : tuple, optional
        which percentiles to apply to the 2p video,
        by default (5,99)

    Returns
    -------
    (generator: generator)
        only if asgenerator == True

    (N_frames: int)
        length of the video in frames, only if asgenerator == True

    (frame_rate: float)
        frame rate of the video, only if asgenerator == True

    Raises
    ------
    FileNotFoundError
        if behaviour video file not found and no individual frames are not found either.
    """
    if dff is None:
        no_dff = True
    else:
        no_dff = False
    if isinstance(select_frames, list) or isinstance(select_frames, np.ndarray) \
        and len(select_frames) == 0 and asgenerator:
        # deal with the case that no frames are selected
        try:
            shape1 = dff.shape[1:]
        except:
            try:
                shape1 = mask.shape
            except:
                dff = get_stack(dff)
                shape1 = dff.shape[1:]
                del dff
        if green is not None:
            try:
                shape2 = green.shape[1:]
            except:
                try:
                    green = get_stack(green)
                    shape2 = green.shape[1:]
                except:
                    shape2=shape1
        else:
            shape2 = (0,0)
        seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_dir)
        images_dir, _ = os.path.split(seven_camera_metadata_file)
        beh_video_dir = os.path.join(images_dir, "camera_{}.mp4".format(camera))  # find_file  *
        beh_generator = generator_video(beh_video_dir)
        shape3, beh_generator = get_generator_shape(beh_generator)
        del beh_generator

        black_image1 = np.zeros((shape1[0], shape1[1], 4), dtype=np.uint8)
        black_image3 = np.zeros(shape3, dtype=np.uint8)
        black_frame_generator1 = utils_video.generators.static_image(black_image1,
                                                                     n_frames=max_length)
        black_frame_generator3 = utils_video.generators.static_image(black_image3,
                                                                     n_frames=max_length)
        if shape2 != (0,0):
            black_image2 = np.zeros((shape2[0], shape2[1], 4), dtype=np.uint8)
            black_frame_generator2 = utils_video.generators.static_image(black_image2,
                                                                         n_frames=max_length)
            generator = utils_video.generators.stack([black_frame_generator3,
                                                      black_frame_generator1,
                                                      black_frame_generator2],
                                                     axis=stack_axis,
                                                     allow_different_length=True)
        else:
            generator = utils_video.generators.stack([black_frame_generator3,
                                                      black_frame_generator1],
                                                     axis=stack_axis,
                                                     allow_different_length=True)
        return generator, np.nan, np.nan


    beh_dir = trial_dir if beh_dir is None else beh_dir
    sync_dir = trial_dir if sync_dir is None else sync_dir
    sync_file = utils2p.find_sync_file(sync_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_dir)
    processed_lines = utils2p.synchronization.get_processed_lines(sync_file, sync_metadata_file,
                                                              metadata_file,
                                                              seven_camera_metadata_file)
    frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"],
                                                             processed_lines["Times"])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"],
                                                              processed_lines["Times"])
    
    dff = get_stack(dff)
    green = get_stack(green)
    red = get_stack(red)
    if green is not None and red is None:
        red = np.zeros_like(green)
    elif green is not None:
        assert green.shape == red.shape

    if no_dff:
        dff = np.zeros_like(green)

    if len(dff) < len(frame_times_2p):
        # adapt for denoised data when some frames in the beginning and in the end are missing
        len_diff = (len(frame_times_2p) - len(dff)) // 2
        first_frame_start_time = frame_times_2p[len_diff]
        last_frame_end_time = frame_times_2p[len_diff+len(dff)]
        frame_times_2p = frame_times_2p[len_diff:len_diff+len(dff)]
        beh_start_ind = np.int(np.argwhere(np.diff(frame_times_beh>first_frame_start_time))) + 1
        try:
            beh_end_ind = np.int(np.argwhere(np.diff(frame_times_beh>last_frame_end_time)))
        except:
            beh_end_ind = len(frame_times_beh)
        frame_times_beh = frame_times_beh[beh_start_ind:beh_end_ind]
    else:
        beh_start_ind = 0
        beh_end_ind = len(frame_times_beh)

    if green is not None:
        if len(green) > len(dff):
            if len(green) >= len(dff) + 2*len_diff:
                green = green[len_diff:len_diff+len(dff), :, :]
                red = red[len_diff:len_diff+len(dff), :, :]
        if green.shape != dff.shape:
            green_y, green_x = green.shape[1:]
            dff_y, dff_x = dff.shape[1:]
            crop_y = (green_y - dff_y) // 2
            crop_x = (green_x - dff_x) // 2
            green = green[:, crop_y:crop_y+dff_y, crop_x:crop_x+dff_x]
            red = red[:, crop_y:crop_y+dff_y, crop_x:crop_x+dff_x]
        # twop_generator = utils_video.generators.frames_2p(red, green, percentiles=(5,99))
        twop_generator = generator_frames_2p(red, green, percentiles=twop_percentiles)
    else:
        twop_generator = None

    if select_frames is not None:
        max_length = len(select_frames) if max_length is None else max_length
        if len(select_frames) < max_length:
            factor = np.ceil(max_length/len(select_frames))
            select_frames = np.repeat(np.expand_dims(select_frames, axis=0),
                                      factor, axis=0).flatten()[:max_length]

    if not no_dff:
        dff_generator = generator_dff(dff, vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax,
                                    blur=blur, mask=mask, crop=crop, colorbarlabel=colorbarlabel,
                                    text=text if text_loc=="dff" else None, log_lim=log_lim)
    else:
        dff_generator = None

    images_dir, _ = os.path.split(seven_camera_metadata_file)
    beh_video_dir = os.path.join(images_dir, f"camera_{camera}.mp4")  # find_file  *
    beh_generator = generator_video(beh_video_dir, start=beh_start_ind,
                                    stop=beh_end_ind, required_n_frames=len(frame_times_beh))

    if time:
        text = [f"{t:.1f}s" for t in frame_times_beh]
        beh_generator = utils_video.generators.add_text(beh_generator, text, scale=3, pos=(10, 150))
    elif text_loc == "beh":
        beh_generator = utils_video.generators.add_text(beh_generator, text, scale=2, pos=(10, 50))
    with open(seven_camera_metadata_file, "r") as f:
        metadata = json.load(f)

    # frame_rate = metadata["FPS"]
    frame_rate = 1 / np.mean(np.diff(frame_times_2p))

    indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)),
                                                        processed_lines["Cameras"],
                                                        processed_lines["Frame Counter"])
    edges = utils2p.synchronization.edges(indices)[0]
    beh_generator = utils_video.generators.resample(beh_generator, edges)
    """
    dff_generator = utils_video.generators.resample(dff_generator, indices)
    """
    if twop_generator is None and dff_generator is not None:
        generator = utils_video.generators.stack([beh_generator, dff_generator],
                                                 axis=stack_axis, allow_different_length=True)
    elif dff_generator is not None:
        # twop_generator = utils_video.generators.resample(twop_generator, indices)
        generator = utils_video.generators.stack([beh_generator, dff_generator, twop_generator],
                                                 axis=stack_axis, allow_different_length=True)
    else:
        generator = utils_video.generators.stack([twop_generator, beh_generator],
                                                 axis=stack_axis, allow_different_length=True)

    if select_frames is not None:
        generator = selected_frames_generator(generator, select_frames)
    if stim_start is not None and stim_stop is not None:
        generator = stimulus_dot_generator(generator, stim_start, stim_stop)
    if downsample is not None and isinstance(downsample, int) and downsample > 1:
        generator = downsample_generator(generator, downsample)
        # N_frames = len(frame_times_beh) // downsample - 1
        N_frames = len(edges) // downsample - 1
        frame_rate = frame_rate / downsample
        if max_length is not None:
            N_frames = max_length // downsample - 1
    else:
        # N_frames = len(frame_times_beh) - 1
        N_frames = len(edges) - 1
        if select_frames is not None:
            N_frames = max_length - 1
    if not asgenerator:
        if not video_name.endswith(".mp4"):
            video_name = video_name + ".mp4"
        make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=int(N_frames))
    else:
        return generator, int(N_frames), frame_rate

def make_multiple_video_raw_dff_beh(dffs, trial_dirs, out_dir, video_name, beh_dirs=None,
                                    sync_dirs=None, camera=6, stack_axes=[0, 1], greens=None,
                                    reds=None, colorbarlabel="dff", vmin=0, vmax=None, pmin=1,
                                    pmax=99, share_lim=True, log_lim=False, blur=0, mask=None,
                                    share_mask=False, crop=None, text=None, text_loc="dff",
                                    downsample=None, select_frames=None, max_length=None, time=True,
                                    stim_starts=None, stim_stops=None, twop_percentiles=(5,99)):
    """
    Make a synchronised and stacked video of (one behavioural camera + dff and/or red/green)
    for multiple trials. Running this function can take very long
    because large amounts of data are involved.

    Parameters
    ----------
    dffs : list of numpy array or str
        for each trial, delta F/F either as numpy array or as absolute path pointing to a .tif

    trial_dirs : list of str
        for each trial, directory containing the 2p data and ThorImage output

    out_dir : str
        directory to save the video in

    video_name : str
        name of the output video

    beh_dirs : list of str, optional
        for each trial, directory containing the 7 camera data. If not specified, will be set equal
        to trial_dir, by default None

    sync_dirs : list of str, optional
        for each trial, directory containing the output of ThorSync. 
        If not specified, will be set equal to trial_dir, by default None

    camera : int, optional
        which camera to use. will search for "camera_X.mp4", by default 6

    stack_axes : list, optional
        how to stack the generators. First value stacks behaviour/dff/... of individual trials,
        second values stacks trials. 0=y, 1=y, by default [0, 1]

    greens : list of (numpy array or str), optional
        for each trial, green stack or path to ,tif, by default None

    reds : list of (numpy array or str), optional
        for each trial, red stack or path to .tif, by default None

    colorbarlabel : str, optional
        name of the dff colourbar label, by default "dff"

    vmin : float, optional
        absolute minimum values for dff, if not specified, use pmin, by default 0

    vmax : float, optional
        absolute maximum values for dff. if not specified, use pmax, by default None

    pmin : float, optional
        percentage min of dff, overruled by vmin by default 1

    pmax : float, optional
        percentage max of dff, overruled by vmax by default 99

    share_lim : bool, optional
        whether to use the same colorscale limits for all trials, by default True

    log_lim : bool, optional
        EXPERIMENTAL FUNCTION: whether to use logarithmic limits and scale for dff,
        by default False

    blur: float, optional
        whether to spatially blur dff. width of Gaussian kernel, by default 0

    mask : numpy array or str or list of the before, optional
        mask to apply over the dff. Either specify list as long as trial_dirs
        or one mask and select share_mask=True, by default None

    share_mask : bool, optional
        whether to share the mask across all trials of dff, by default False

    crop : list, optional
        cropping applied to dff
        list of length 2 for symmetric cropping (same on both sides),
        or list of length 4 for assymetric cropping, by default None

    text : list of str, optional
        for each trial, string to write on top of the video, by default None

    text_loc : str, optional
        where the text will be displayed. either "dff" or "beh".
        If "beh", make sure to select time=False, by default "dff"

    downsample : int, optional
        downsampling factor, by default None

    select_frames : list of (list or numpy array), optional
        for each trial, list containing a sequence of selected two-photon frame inidices.
        if an empty list, generate a black frame. If None, use all frames, by default None

    max_length : int, optional
        maximum length of the video in frames, by default None

    time : bool, optional
        whether or not to show the current time of the recording in the behaviour video,
        by default True

    twop_percentiles : tuple, optional
        which percentiles to apply to the 2p video,
        by default (5,99)
    """
    if not isinstance(dffs, list):
        dffs = [dffs]
    if text is not None:
        if not isinstance(text, list):
            text = [text]
        assert len(text) == len(dffs)
    else:
        text = [None for _ in  dffs]
    if share_mask or mask is None:
        mask = [mask for _ in dffs]
    assert len(mask) == len(dffs)
    if greens is not None:
        if not isinstance(greens, list):
            greens = [greens]
        assert len(greens) == len(dffs)
    else:
        greens = [None for _ in dffs]
    if reds is not None:
        if not isinstance(reds, list):
            reds = [reds]
        assert len(reds) == len(dffs)
    else:
        reds = [None for _ in dffs]
    
    dffs = [get_stack(dff) for dff  in dffs]
    dffs_tmp = []
    for this_mask, this_dff in zip(mask, dffs):
        if this_mask is not None and this_dff is not None:
            this_mask = np.invert(this_mask)  # invert only once
            this_dff_tmp = np.zeros_like(this_dff)
            for i_f, f in enumerate(this_dff):
                f[this_mask] = 0
                this_dff_tmp[i_f, :, :] = f
            dffs_tmp.append(this_dff_tmp)
        else:
            dffs_tmp.append(this_dff)
    dffs = dffs_tmp
    
    if share_lim:
        vmins = [np.percentile(dff, pmin) if vmin is None and dff is not None else vmin for dff in dffs]
        vmaxs = [np.percentile(dff, pmax) if vmax is None and dff is not None else vmax for dff in dffs]
        vmin = np.ma.average(np.ma.array(vmins)) if not all([v is None for v in vmins]) else None
        vmax = np.ma.average(np.ma.array(vmaxs)) if not all([v is None for v in vmaxs]) else None
    if isinstance(log_lim, bool): #TODO: test this
        log_lim = [log_lim for _ in dffs]
    assert len(dffs) == len(log_lim)

    assert len(dffs) == len(trial_dirs)
    if beh_dirs is not None:
        if not isinstance(beh_dirs, list):
            beh_dirs = [beh_dirs]
        assert len(beh_dirs) == len(dffs)
    else:
        beh_dirs = [None for _ in  dffs]
    if sync_dirs is not None:
        if not isinstance(sync_dirs, list):
            sync_dirs = [sync_dirs]
        assert len(sync_dirs) == len(dffs)
    else:
        sync_dirs = [None for _ in  dffs]
    if select_frames is not None:
        assert len(select_frames) == len(dffs)
        tmp_max_length = np.max([len(frames) for frames in select_frames])
        max_length = tmp_max_length if max_length is None else np.min([max_length, tmp_max_length])
    else:
        select_frames = [None for _ in  dffs]

    if stim_starts is not None:
        assert len(stim_starts) == len(dffs)
    else:
        stim_starts = [None for _ in dffs]
    if stim_stops is not None:
        assert len(stim_stops) == len(dffs)
    else:
        stim_stops = [None for _ in dffs]

    generators = []
    N_frames = []
    frame_rates = []
    for i_gen, (dff, trial_dir, beh_dir, sync_dir, this_text, this_mask, \
        green, red, frames, this_log_lim, stim_start, stim_stop) \
        in enumerate(zip(dffs, trial_dirs, beh_dirs, sync_dirs, text, mask, \
            greens, reds, select_frames, log_lim, stim_starts, stim_stops)):
        this_generator, this_N_frames, frame_rate = make_video_raw_dff_beh(
            dff=dff, trial_dir=trial_dir, out_dir=None, video_name=None,
            beh_dir=beh_dir, sync_dir=sync_dir, camera=camera, stack_axis=stack_axes[0],
            green=green, red=red, colorbarlabel=colorbarlabel,
            vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, blur=blur, mask=this_mask,
            crop=crop, text=this_text, text_loc=text_loc, log_lim = this_log_lim, time=time,
            asgenerator=True, downsample=downsample, max_length=max_length, select_frames=frames,
            stim_start=stim_start, stim_stop=stim_stop, twop_percentiles=twop_percentiles)
        generators.append(this_generator)
        N_frames.append(this_N_frames)
        frame_rates.append(frame_rate)
    if not len(np.unique(frame_rates)) == 1:
        print("Frame rates are: ", frame_rates)
        frame_rate = np.nanmean(frame_rates)
    N_frames = int(np.nanmin(N_frames))
    generator = utils_video.generators.stack(generators, axis=stack_axes[1],
                                             allow_different_length=True)
    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)

def stimulus_dot_generator(generator, start_stim, stop_stim):
    """add a red dot to a video generator whenever a stimulus was on.
    Will add red dot upon start_stim and remove it at stop_stim

    Parameters
    ----------
    generator : generator
    
    start_stim : list
        list of indices when the stimulation started
    
    stop_stim : list
        list of indices when the stimulation ended

    Yields
    -------
    frame: numpy array
    """
    if not isinstance(start_stim, list) and not isinstance(start_stim, tuple):
        start_stim = [start_stim]
    if not isinstance(stop_stim, list) and not isinstance(stop_stim, tuple):
        stop_stim = [stop_stim]
    stim_status = False
    for i_frame, frame in enumerate(generator):
        if i_frame in start_stim:
            stim_status = True
        elif i_frame in stop_stim:
            stim_status = False
        if stim_status:
            im_size = frame.shape[0]
            factor = im_size / 480
            frame = frame.copy()
            cv2.circle(frame, (int(50*factor),int(50*factor)), int(40*factor), (255,0,0), -1)
        yield frame

def make_behaviour_grid_video(video_dirs, start_frames, N_frames, stim_range, out_dir, video_name, frame_rate=None, size=None):
    # video_dirs = [fly1, fly2]
    # start_frames = [[ind1, ind2], [ind3, ind4]]
    # N_frames = 100*20
    # stim_range = [500,1500]
    assert all(np.array(stim_range) < N_frames)
    generators = []
    frame_rates = []
    for i_fly, (video_dir, start_frames_list) in enumerate(zip(video_dirs, start_frames)):
        metadata_dir = utils2p.find_seven_camera_metadata_file(os.path.dirname(video_dir))
        with open(metadata_dir, "r") as f:
            metadata = json.load(f)
        frame_rates.append(metadata["FPS"])
        del metadata

        for start_frame in start_frames_list:
            this_generator = generator_video(video_dir, start=start_frame, stop=start_frame+N_frames, size=size)
            this_generator = stimulus_dot_generator(this_generator, stim_range[0], stim_range[1])
            generators.append(this_generator)

    mean_frame_rate = np.mean(frame_rates)
    if frame_rate is None:
        frame_rate = mean_frame_rate
    elif not np.isclose(frame_rate, mean_frame_rate):
        print(f"Selected frame rate {frame_rate} is not close to average frame rate of videos {mean_frame_rate}.")

    generator = utils_video.generators.grid(generators)
    if not video_name.endswith(".mp4"):
            video_name = video_name + ".mp4"
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)

def make_all_odour_condition_videos(video_dirs, paradigm_dirs, out_dir, video_name, frame_range=[-500,1500], stim_length=1000,
                                    frame_rate=None, size=None, conditions=None, selected_stims=None):
    assert len(video_dirs) == len(paradigm_dirs)
    if video_name.endswith(".mp4"):
        video_name = video_name[:-4]
    unique_conditions = []
    all_conditions = []
    all_starts = []
    for paradigm_dir in paradigm_dirs:
        with open(paradigm_dir, "rb") as f:
            paradigm = pickle.load(f)
        trial_conditions = paradigm["condition_list"]
        trial_starts = paradigm["start_cam_frames"]
        all_conditions.append(trial_conditions)
        all_starts.append(trial_starts)
        this_conditions = np.unique(trial_conditions)
        unique_conditions = np.concatenate((unique_conditions, this_conditions))
    unique_conditions = np.unique(unique_conditions)

    if conditions is not None:
        unique_conditions = [this_cond for this_cond in unique_conditions if this_cond in conditions]
    
    if selected_stims is None:
        selected_stims = [None for _ in paradigm_dirs]

    for condition in unique_conditions:
        start_frames = []
        for trial_conditions, trial_starts, selected_trial_stims in zip(all_conditions, all_starts, selected_stims):
            trial_start_frames = []
            i_stim = 0
            for this_condition, this_start in zip(trial_conditions, trial_starts):
                if this_condition == condition:
                    if selected_trial_stims is None or i_stim in selected_trial_stims:
                        trial_start_frames.append(this_start+frame_range[0])
                    i_stim += 1
            start_frames.append(trial_start_frames)

        this_video_name = video_name + "_" + condition
        make_behaviour_grid_video(video_dirs, start_frames=start_frames, N_frames=frame_range[1]-frame_range[0],
                                  stim_range=[-frame_range[0], -frame_range[0]+stim_length], size=size,
                                  out_dir=out_dir, video_name=this_video_name, frame_rate=frame_rate)

def make_2p_grid_video(greens, reds, out_dir, video_name, percentiles=(5,99), frame_rate=None, trial_dir=None, texts=None, force_N_frames=None):
    assert len(greens) == len(reds)
    greens = [get_stack(green) for green in greens]
    reds = [get_stack(red) for red in reds]
    lens_green = [len(green) if green is not None else 0 for green in greens]
    lens_red = [len(red) if red is not None else 0 for red in reds]
    lens_unique = np.unique([lens_green, lens_red])
    N_frames = lens_unique[0] if lens_unique[0] != 0 else lens_unique[1]
    size = greens[0].shape[1:]
    green_perc_low = []
    green_perc_high = []
    red_perc_low = []
    red_perc_high = []
    for i_vid, (green, red) in enumerate(zip(greens, reds)):
        if green is None:
            greens[i_vid] = np.zeros((N_frames, size[0], size[1]))
        elif len(green) > N_frames:
            diff = len(green) - N_frames
            print(f"Difference in frame length of {diff}. Will cut half of it at the front and half of it at the back.")
            shift = diff // 2
            greens[i_vid] = green[shift:shift+N_frames, :, :]
        if green is not None:
            green_perc_low.append(np.percentile(greens[i_vid], percentiles[0]))
            green_perc_high.append(np.percentile(greens[i_vid], percentiles[1]))
        if red is None:
            reds[i_vid] = np.zeros((N_frames, size[0], size[1]))
        elif len(red) > N_frames:
            diff = len(red) - N_frames
            print(f"Difference in frame length of {diff}. Will cut half of it at the front and half of it at the back.")
            shift = diff // 2
            reds[i_vid] = red[shift:shift+N_frames, :, :]
        if red is not None:
            red_perc_low.append(np.percentile(reds[i_vid], percentiles[0]))
            red_perc_high.append(np.percentile(reds[i_vid], percentiles[1]))
    green_perc_low = np.maximum(0, np.mean(green_perc_low))
    green_perc_high = np.maximum(0, np.mean(green_perc_high))
    red_perc_low = np.maximum(0, np.mean(red_perc_low))
    red_perc_high = np.maximum(0, np.mean(red_perc_high))

    generators = []
    for i_vid, (green, red) in enumerate(zip(greens, reds)):
        generator = generator_frames_2p(red_stack=red, green_stack=green, 
                                        red_vlim=(red_perc_low, red_perc_high), 
                                        green_vlim=(green_perc_low, green_perc_high))
        generators.append(generator)
    generators_text = []
    if texts is not None and len(texts) == len(generators):
        for i_vid, (text, generator) in enumerate(zip(texts, generators)):
            generators_text.append(utils_video.generators.add_text(generator, text=text, pos=(10,50)))

    generator = utils_video.generators.grid(generators_text)

    if frame_rate is None and not trial_dir is None:
        meta_data = utils2p.Metadata(utils2p.find_metadata_file(trial_dir))
        frame_rate = meta_data.get_frame_rate()
    elif frame_rate is None:
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    
    if not video_name.endswith(".mp4"):
            video_name = video_name + ".mp4"
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, 
               n_frames=N_frames if force_N_frames is None else force_N_frames)