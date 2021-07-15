# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import json
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

# from torch import from_numpy

import utils2p
import utils2p.synchronization
import utils_video.generators
from utils_video import make_video
from utils_video.utils import resize_shape, colorbar, add_colorbar
from deepfly.CameraNetwork import CameraNetwork

FILE_PATH = os.path.realpath(__file__)
PLOT_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(PLOT_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.utils import get_stack, find_file, crop_img, crop_stack
from longterm import load
from longterm.register.warping import apply_motion_field, apply_offset

def make_video(video_path, frame_generator, fps, output_shape=(-1, 2880), n_frames=-1):
    if np.log2(fps) % 1 == 0:
        fps += 0.01
    utils_video.make_video(video_path, frame_generator, fps, output_shape=output_shape, n_frames=n_frames)

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
        raise NotImplementedError("You have to supply either a valid trial_dir or specify the frame_rate.")

    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"
        
    generator = utils_video.generators.frames_2p(red, green, percentiles=percentiles)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)
    
def generator_dff(stack, size=None, font_size=16, pmin=0.5, pmax=99.5, vmin=None, vmax=None, 
                  blur=0, mask=None, crop=None,
                  text=None):                           
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

    generators = [utils_video.generators.frames_2p(red, green, percentiles=percentiles) for green, red in zip(greens, reds)]
    generator = utils_video.generators.stack(generators, axis=1)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)

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
    generators = [[generator_dff(dff, vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, 
                                 blur=blur, crop=crop, mask=None, text=t) 
                   for dff, t in zip(dffs_, text_)] 
                  for dffs_, text_ in zip(dffs, text)]
    generator_rows = [utils_video.generators.stack(generator_row, axis=1) for generator_row in generators]
    generator = utils_video.generators.stack(generator_rows, axis=0)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate)

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
    N_frames = camNet.cam_list[0].points2d.shape[0]
    camNet.num_images = N_frames

    cmap = plt.cm.get_cmap("seismic")
    i_cs = np.linspace(start=0, stop=1, num=10)
    i_cs[5:] = np.flip(i_cs[5:])
    colors = [np.array(cmap(i_c))*255 for i_c in i_cs]
    if os.path.isfile(os.path.join(image_folder, "camera_{}_img_0.jpg".format(cameras[0]))):
        saved_as_videos = False
    else:
        saved_as_videos = True
        caps = [cv2.VideoCapture(os.path.join(image_folder, "camera_{}.mp4".format(cam))) for cam in cameras]
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
            img = videos[i_cam][i_frame] if saved_as_videos else None
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
            metadata_dir = utils2p.find_seven_camera_metadata_file(trial_dir)
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

def make_video_df3d(trial_dir, out_dir, video_name, frames=None, frame_rate=None, cameras=[5,3,1], 
                    print_frame_num=True, print_frame_time=True, print_beh_label=False, beh_label_dir=None,
                    downsample=None, speedup=1):
    image_dir = os.path.join(trial_dir, "behData", "images")
    if frame_rate is None:
        metadata_dir = utils2p.find_seven_camera_metadata_file(trial_dir)
        with open(metadata_dir, "r") as f:
            metadata = json.load(f)
        frame_rate = metadata["FPS"]
        del metadata

    factor_downsample = 1 if downsample is None else downsample
    
    if not video_name.endswith(".mp4"):
        video_name = video_name + ".mp4"

    generator = generator_df3d(image_dir, cameras=cameras, frame_rate=frame_rate, N_frames=frames,
                               print_frame_time=print_frame_time, print_frame_num=print_frame_num,
                               print_beh_label=print_beh_label, beh_label_dir=beh_label_dir,
                               factor_downsample=factor_downsample, speedup=speedup)
    make_video(os.path.join(out_dir, video_name), generator, frame_rate/factor_downsample*speedup)

def generator_video(path, size=None, start=0, stop=9223372036854775807):
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

def downsample_generator(generator, factor):
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
    i_frame = 0
    except_frame = None
    frames = []
    first_except = True
    while 1:
        try:
            frame = next(generator)
            if i_frame == 0:
                except_frame = np.zeros_like(frame)
            if i_frame not in select_frames:
                i_frame += 1
                continue
            else:
                i_frame += 1
                frames.append(frame)
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
                           camera=6, stack_axis=0, green=None, red=None,
                           vmin=0, vmax=None, pmin=1, pmax=99, blur=0, mask=None, crop=None, 
                           text=None, asgenerator=False, downsample=None, select_frames=None, max_length=None):
    beh_dir = trial_dir if beh_dir is None else beh_dir
    sync_dir = trial_dir if sync_dir is None else sync_dir
    sync_file = utils2p.find_sync_file(sync_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_dir)
    processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"], processed_lines["Times"])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"], processed_lines["Times"])
    
    dff = get_stack(dff)
    green = get_stack(green)
    red = get_stack(red)
    if green is not None and red is None:
        red = np.zeros_like(green)
    elif green is not None:
        assert green.shape == red.shape

    if len(dff) < len(frame_times_2p):  # adapt for denoised data when some frames in the beginning and some frames in the end are missing
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
        twop_generator = utils_video.generators.frames_2p(red, green, percentiles=(5,99))
    else:
        twop_generator = None

    if select_frames is not None:
        max_length = len(select_frames) if max_length is None else max_length
        if len(select_frames) < max_length:
            factor = np.ceil(max_length/len(select_frames))
            select_frames = np.repeat(np.expand_dims(select_frames, axis=0),factor, axis=0).flatten()[:max_length]

    dff_generator = generator_dff(dff, vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, 
                                  blur=blur, mask=mask, crop=crop, text=text)

    images_dir, _ = os.path.split(seven_camera_metadata_file)
    # beh_video_dir = os.path.join(images_dir, "camera_{}.mp4".format(camera))
    beh_video_dir = find_file(images_dir, "camera_{}*.mp4".format(camera))
    beh_generator = generator_video(beh_video_dir, start=beh_start_ind, stop=beh_end_ind)  # TODO: selected frames
    text = [f"{t:.1f}s" for t in frame_times_beh]
    beh_generator = utils_video.generators.add_text(beh_generator, text, scale=3, pos=(10, 150))
    with open(seven_camera_metadata_file, "r") as f:
        metadata = json.load(f)

    # frame_rate = metadata["FPS"]
    frame_rate = 1 / np.mean(np.diff(frame_times_2p))

    indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)), processed_lines["Cameras"], processed_lines["Frame Counter"])
    edges = utils2p.synchronization.edges(indices)[0]
    beh_generator = utils_video.generators.resample(beh_generator, edges)
    """
    dff_generator = utils_video.generators.resample(dff_generator, indices)
    """
    if twop_generator is None:
        generator = utils_video.generators.stack([beh_generator, dff_generator], axis=stack_axis)
    else:
        # twop_generator = utils_video.generators.resample(twop_generator, indices)
        generator = utils_video.generators.stack([beh_generator, dff_generator, twop_generator], axis=stack_axis)
    
    if select_frames is not None:
        generator = selected_frames_generator(generator, select_frames)
    if downsample is not None and isinstance(downsample, int) and downsample > 1:
        generator = downsample_generator(generator, downsample)
        N_frames = len(frame_times_beh) // downsample - 1
        frame_rate = frame_rate / downsample
        if select_frames is not None:
            N_frames = max_length // downsample - 1
    else:
        N_frames = len(frame_times_beh) - 1
        if select_frames is not None:
            N_frames = max_length - 1
    if not asgenerator:
        if not video_name.endswith(".mp4"):
            video_name = video_name + ".mp4"
        make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)
    else:
        return generator, N_frames, frame_rate

def make_multiple_video_raw_dff_beh(dffs, trial_dirs, out_dir, video_name, beh_dirs=None, sync_dirs=None, 
                                    camera=6, stack_axes=[0, 1], greens=None, reds=None,
                                    vmin=0, vmax=None, pmin=1, pmax=99, share_lim=True, 
                                    blur=0, mask=None, share_mask=False, crop=None, text=None, 
                                    downsample=None, select_frames=None):
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
        if this_mask is not None:
            this_mask = np.invert(this_mask)  # invert only once
            this_dff_tmp = np.zeros_like(this_dff)
            for i_f, f in enumerate(this_dff):
                f[this_mask] = 0
                this_dff_tmp[i_f, :, :] = f
            dffs_tmp.append(this_dff_tmp)
        else:
            dffs_tmp.append(this_dff)
    dffs = dffs_tmp

    # reds = [get_stack(red) for red  in reds]
    # greens = [get_stack(green) for green  in greens]
    
    if share_lim:
        vmins = [np.percentile(dff, pmin) if vmin is None else vmin for dff in dffs]
        vmaxs = [np.percentile(dff, pmax) if vmax is None else vmax for dff in dffs]
        vmin = np.mean(vmins)
        vmax = np.mean(vmaxs)

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
        max_length = np.max([len(frames) for frames in select_frames])
    else:
        select_frames = [None for _ in  dffs]
        max_length = None

    generators = []
    N_frames = []
    frame_rates = []
    for i_gen, (dff, trial_dir, beh_dir, sync_dir, this_text, this_mask, green, red, frames) \
        in enumerate(zip(dffs, trial_dirs, beh_dirs, sync_dirs, text, mask, greens, reds, select_frames)):
        this_generator, this_N_frames, frame_rate = make_video_raw_dff_beh(dff=dff, trial_dir=trial_dir, out_dir=None, video_name=None,
                                                                  beh_dir=beh_dir, sync_dir=sync_dir, camera=camera, stack_axis=stack_axes[0],
                                                                  green=green, red=red,
                                                                  vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, blur=blur, mask=this_mask,
                                                                  crop=crop, text=this_text, 
                                                                  asgenerator=True, downsample=downsample, max_length=max_length, select_frames=frames)
        generators.append(this_generator)
        N_frames.append(this_N_frames)
        frame_rates.append(frame_rate)
    if not len(np.unique(frame_rates)) == 1:
        print("Frame rates are: ", frame_rates)
        frame_rate = np.mean(frame_rates)
    N_frames = np.min(N_frames)
    generator = utils_video.generators.stack(generators, axis=stack_axes[1])
    if not video_name.endswith(".mp4"):
            video_name = video_name + ".mp4"
    make_video(os.path.join(out_dir, video_name), generator, frame_rate, n_frames=N_frames)



if __name__ == "__main__":

    JB_DATA = False
    LH_DATA = False
    DENOISED = False
    PIPELINE = False
    DF3D = True
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

    elif DF3D:
        date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")  # 210216_J1xCI9 fly1 trial 0
        fly_dirs = load.get_flies_from_datedir(date_dir)
        trial_dirs = load.get_trials_from_fly(fly_dirs)[0]
        for trial_dir in trial_dirs:
            print(trial_dir)
            # out_dir=os.path.join(trial_dir, "behData", "images", "df3d")
            out_dir = os.path.join(trial_dir, "behData")
            video_name = "beh_classify"  # "df3d"
            make_video_df3d(trial_dir, out_dir,
                            print_beh_label=True,
                            beh_label_dir=os.path.join(trial_dir, load.PROCESSED_FOLDER, "behaviour_labels.pkl"),
                            video_name=video_name, cameras=[5], downsample=1, speedup=1)  # [5, 1], 5, 10
        pass