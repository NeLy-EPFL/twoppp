# Jonas Braun
# jonas.braun@epfl.ch
# 18.02.2021

import os
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
import cv2

import utils2p
import ofco

FILE_PATH = os.path.realpath(__file__)
REGISTER_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(REGISTER_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils

def save_ref_frame(stack, ref_frame_dir, i_frame = 0, com_pre_reg=True, overwrite=False):
    if os.path.isfile(ref_frame_dir):
        return
    stack = utils.get_stack(stack)
    ref_frame = stack[i_frame:i_frame+1, :, :]
    if com_pre_reg:
        ref_frame = center_stack(ref_frame, return_offset=False)

    utils2p.save_img(ref_frame_dir, ref_frame)

def center_stack(frames, return_offset=False, sigma_filt=(0, 10, 10), foreground_thres=0.75):
    N, H, W = frames.shape
    offsets = np.zeros((N,2))
    frames_filt = gaussian_filter(frames, sigma=sigma_filt, truncate=2)
    for i_frame, (frame, frame_filt) in enumerate(zip(frames, frames_filt)):
        thres = np.quantile(frame_filt, foreground_thres)
        frame_thres = (frame_filt > thres).astype(np.float)
        offsets[i_frame, :] = np.array(center_of_mass(frame_thres))  - np.array([H/2, W/2])
    offsets = offsets.astype(np.int)
    frames_shifted = apply_offset(frames, offsets)
    if return_offset:
        return frames_shifted, offsets
    else:
        return frames_shifted

def apply_offset(frames, offsets):
    if frames.ndim == 2:  # TODO: test this
        frames = frames[np.newaxis, :, :]
        squeeze = True
    else:
        squeeze = False
    dtype = frames.dtype
    N, H, W = frames.shape
    border_value = np.mean(frames[:, :10,:10])
    frames_shifted = np.ones_like(frames) * border_value
    for i_frame, (frame, offset) in enumerate(zip(frames, offsets)):
        T = np.float32([[1,0, -offset[1]], [0, 1, -offset[0]]])
        frames_shifted[i_frame, :, :] = cv2.warpAffine(frame, T, (W,H), 
                                                       borderMode=cv2.BORDER_CONSTANT, 
                                                       borderValue=border_value.astype(np.float))

    if squeeze:
        frames_shifted = np.squeeze(frames_shifted)
    return frames_shifted.astype(dtype)

def warp(stack1, stack2=None, ref_frame=None, stack1_out_dir=None, stack2_out_dir=None, 
         com_pre_reg=False, offset_dir=None, return_stacks=False, overwrite=False,
         select_frames=None, parallel=True, verbose=False, w_output=None, initial_w=None, param=None):
    param = ofco.utils.default_parameters() if param is None else param

    if os.path.isfile(stack1_out_dir) and not overwrite:
        if not return_stacks:
            return None, None
        stack1_warped = utils2p.load_img(stack1_out_dir)
        if stack2 is not None and os.path.isfile(stack2_out_dir):
            stack2_warped = utils2p.load_img(stack2_out_dir)
        else:
            stack2_warped = None
        return stack1_warped, stack2_warped

    if isinstance(stack1, str):
        stack1 = utils2p.load_img(stack1)
        if com_pre_reg and os.path.isfile(offset_dir):
            offsets = np.load(offset_dir)
            stack1 = apply_offset(stack1, offsets)
        elif com_pre_reg:
            stack1, offsets = center_stack(stack1, return_offset=True)
            np.save(offset_dir, offsets)
    if not isinstance(stack1, np.ndarray):
        raise NotImplementedError

    N_frames, N_y, N_x = stack1.shape

    if stack2 is not None:
        stack2 = utils.get_stack(stack2)
        assert(stack1.shape == stack2.shape)
        if com_pre_reg:
            stack2 = apply_offset(stack2, offsets)

    if isinstance(ref_frame, int):
        assert ref_frame < N_frames
    elif ref_frame is not None:
        ref_frame = utils.get_stack(ref_frame)
        ref_frame = np.squeeze(ref_frame)
        if ref_frame.ndim == 3:
            ref_frame = ref_frame[0, :, :]
        assert(ref_frame.shape == (N_y, N_x))
    else:
        ref_frame = 0

    stack1_warped, stack2_warped = ofco.motion_compensate(stack1, stack2, 
                                                          frames=range(N_frames) if select_frames is None else select_frames,
                                                          param=param, verbose=verbose, parallel=parallel, w_output=w_output, 
                                                          initial_w=initial_w, ref_frame=ref_frame)

    if stack1_out_dir is not None:
        utils2p.save_img(stack1_out_dir, stack1_warped)
    if stack2_out_dir is not None:
        utils2p.save_img(stack2_out_dir, stack2_warped)

    return stack1_warped, stack2_warped if return_stacks else None, None

def warp_N_parts(stack1, stack1_out_dir, N_parts, stack2=None, stack2_out_dir=None, ref_frame=None, 
         com_pre_reg=False, offset_dir=None, return_stacks=False, overwrite=False,
         select_frames=None, parallel=True, verbose=False, w_output=None, initial_w=None, param=None):
    param = ofco.utils.default_parameters() if param is None else param

    if os.path.isfile(stack1_out_dir) and not overwrite:
        if not return_stacks:
            return None, None
        stack1_warped = utils2p.load_img(stack1_out_dir)
        if stack2 is not None and os.path.isfile(stack2_out_dir):
            stack2_warped = utils2p.load_img(stack2_out_dir)
        else:
            stack2_warped = None
        return stack1_warped, stack2_warped

    if isinstance(stack1, str):
        stack1 = utils2p.load_img(stack1)
        if com_pre_reg and os.path.isfile(offset_dir):
            offsets = np.load(offset_dir)
            stack1 = apply_offset(stack1, offsets)
        elif com_pre_reg:
            stack1, offsets = center_stack(stack1, return_offset=True)
            np.save(offset_dir, offsets)
    if not isinstance(stack1, np.ndarray):
        raise NotImplementedError

    N_frames, N_y, N_x = stack1.shape

    if stack2 is not None:
        stack2 = utils.get_stack(stack2)
        assert(stack1.shape == stack2.shape)
        if com_pre_reg:
            stack2 = apply_offset(stack2, offsets)

    if isinstance(ref_frame, int):
        assert ref_frame < N_frames
    elif ref_frame is not None:
        ref_frame = utils.get_stack(ref_frame)
        ref_frame = np.squeeze(ref_frame)
        if ref_frame.ndim == 3:
            ref_frame = ref_frame[0, :, :]
    else:
        ref_frame = stack1[0, :, :]
    assert(ref_frame.shape == (N_y, N_x))

    N_per_part = np.ceil(N_frames / N_parts)
    for i_part in np.arange(N_parts):
        ind_part = np.arange(i_part*N_per_part, np.min((N_frames, (i_part+1)*N_per_part)))
        # TODO: incorporate select_frames and initial_w
        w_output_i = w_output[:-4] + "_{}".format(i_part) + w_output[-4:] if not w_output is None else None

        stack1_part_warped, stack2_part_warped = ofco.motion_compensate(stack1[ind_part, :, :], stack2[ind_part, :, :] if stack2 is not None else None, 
                                                          frames=range(len(ind_part)) if select_frames is None else select_frames,
                                                          param=param, verbose=verbose, parallel=parallel, w_output=w_output_i, 
                                                          initial_w=initial_w, ref_frame=ref_frame)

        if i_part == 0:
            utils2p.save_img(stack1_out_dir, stack1_part_warped)
            if stack2_part_warped is not None:
                utils2p.save_img(stack2_out_dir, stack2_part_warped)
        else:
            stack1_warped = utils2p.load_img(stack1_out_dir)
            stack1_warped = np.concatenate((stack1_warped, stack1_part_warped), axis=0)
            utils2p.save_img(stack1_out_dir, stack1_warped)
            if i_part < N_parts -  1:
                del stack1_warped
            if stack2_part_warped is not None:
                stack2_warped = utils2p.load_img(stack2_out_dir)
                stack2_warped = np.concatenate((stack2_warped, stack2_part_warped), axis=0)
                utils2p.save_img(stack2_out_dir, stack2_warped)
                if i_part < N_parts -  1:
                    del stack2_warped
        del stack1_part_warped
        del stack2_part_warped

    return stack1_warped, stack2_warped if return_stacks else None, None
 

def apply_warp(stack, stack_out_dir, w_output, select_frames=None,
               com_pre_reg=False, offset_dir=None, return_stacks=False, overwrite=False):
    if os.path.isfile(stack_out_dir) and not overwrite:
        if not return_stacks:
            return None
        stack_warped = utils2p.load_img(stack_out_dir)
        return stack_warped

    if isinstance(stack, str):
        stack = utils2p.load_img(stack)
        if com_pre_reg and os.path.isfile(offset_dir):
            offsets = np.load(offset_dir)
            stack = apply_offset(stack, offsets)
        elif com_pre_reg:
            stack, offsets = center_stack(stack, return_offset=True)
            np.save(offset_dir, offsets)
    if not isinstance(stack, np.ndarray):
        raise NotImplementedError

    if select_frames is not None:
        stack = stack[select_frames, :, :]

    N_frames, N_y, N_x = stack.shape

    motion_field = np.load(w_output)
    assert motion_field.shape == (N_frames, N_y, N_x, 2)

    stack_warped = apply_motion_field(stack, motion_field)

    if stack_out_dir is not None:
        utils2p.save_img(stack_out_dir, stack_warped)

    return stack_warped if return_stacks else None

def apply_motion_field(stack, motion_field):
    N_frames, N_y, N_x = stack.shape
    assert motion_field.shape == (N_frames, N_y, N_x, 2)

    stack_warped = np.zeros_like(stack)

    for i_frame in range(N_frames):
        stack_warped[i_frame, :, :] = ofco.warping.bilinear_interpolate(stack[i_frame, :, :], 
                                                                        motion_field[i_frame, :, :, 0], 
                                                                        motion_field[i_frame, :, :, 1])
    
    return stack_warped


if __name__ == "__main__":

    COMPUTE = True
    APPLY = False


    date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    all_trial_dirs = load.get_trials_from_fly(fly_dirs)

    for trial_dirs, fly_dir in zip(all_trial_dirs, fly_dirs):
        ref_stack = os.path.join(trial_dirs[0], load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
        ref_frame_dir = os.path.join(fly_dir, load.PROCESSED_FOLDER, "ref_frame_com.tif")  # ref_frame_com, ref_frame
        save_ref_frame(ref_stack, ref_frame_dir=ref_frame_dir, com_pre_reg=True)  # True, False
        
        for trial_dir in trial_dirs:
            print("processing trial: " + trial_dir)
            processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)

            stack1 = os.path.join(processed_dir, load.RAW_RED_TIFF)
            stack2 = os.path.join(processed_dir, load.RAW_GREEN_TIFF)
            stack1_out_dir = os.path.join(processed_dir, "red_com_warped.tif")  # red_com_warped, red_warped
            stack2_out_dir = os.path.join(processed_dir, "green_com_warped.tif")  # green_com_warped, green_warped
            w_output = os.path.join(processed_dir, "motion_field_com.npy")  # motion_field_com, motion_field

            offset_dir = os.path.join(processed_dir, "com_offset.npy")
            if COMPUTE:
                warp(stack1=stack1, stack2=stack2, ref_frame=ref_frame_dir, 
                    stack1_out_dir=stack1_out_dir, stack2_out_dir=stack2_out_dir, 
                    com_pre_reg=True, offset_dir=offset_dir,  # True, False
                    return_stacks=False, overwrite=True, select_frames=None,
                    parallel=True, verbose=True, w_output=w_output)
            
            if APPLY:
                apply_warp(stack=stack2, stack_out_dir=stack2_out_dir, w_output=w_output, 
                           com_pre_reg=True, offset_dir=offset_dir,   # True, False
                           return_stacks=False, overwrite=True, select_frames=None)

        pass