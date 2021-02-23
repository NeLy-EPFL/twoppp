# Jonas Braun
# jonas.braun@epfl.ch
# 18.02.2021

import os
import sys
import numpy as np

import utils2p
import ofco

FILE_PATH = os.path.realpath(__file__)
REGISTER_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(REGISTER_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils
from longterm.register.deepreg_prepare import center_stack, apply_offset

def warp(stack1, stack2=None, ref_stack=None, stack1_out_dir=None, stack2_out_dir=None, 
         com_pre_reg=False, offset_dir=None, ref_offset_dir=None, return_stacks=False, overwrite=False,
         select_frames=None, parallel=True, verbose=False, w_output=None, initial_w=None, param=ofco.utils.default_parameters()):
    
    if os.path.isfile(stack1_out_dir) and not overwrite:
        if not return_stacks:
            return None, None
        stack1_warped = utils2p.load_img(stack1_out_dir)
        if stack2 is not None and os.path.isfile(stack2_out_dir):
            stack2_warped = utils2p.load_img(stack2_out_dir)
        else:
            stack2_warped = None
        return 

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

    N_frames, N_x, N_y = stack1.shape

    if stack2 is not None:
        stack2 = utils.get_stack(stack2)
        assert(stack1.shape == stack2.shape)
        if com_pre_reg:
            stack2 = apply_offset(stack2, offsets)


    if ref_stack is not None:
        ref_stack = utils.get_stack(ref_stack)
        
        if com_pre_reg and os.path.isfile(ref_offset_dir):
            ref_offsets = np.load(offset_dir)
            ref_stack = apply_offset(ref_stack, ref_offsets)
            ref_frame = ref_stack[0, :, :]
        elif com_pre_reg:
            ref_stack = center_stack(ref_stack[0, :, :], return_offset=False)
        
        assert(ref_frame.shape == (N_x, N_y))
    else:
        ref_frame = 0
    del ref_stack

    stack1_warped, stack2_warped = ofco.motion_compensate(stack1, stack2, 
                                                          frames=range(N_frames) if select_frames is None else select_frames,
                                                          param=param, verbose=verbose, parallel=parallel, w_output=w_output, 
                                                          initial_w=initial_w, ref_frame=ref_frame)

    if stack1_out_dir is not None:
        utils2p.save_img(stack1_out_dir, stack1_warped)
    if stack2_out_dir is not None:
        utils2p.save_img(stack2_out_dir, stack1_warped)

    if return_stacks:
        return stack1_warped, stack2_warped
    else:
        return None, None


if __name__ == "__main__":
    date_dir = os.path.join(load.LOCAL_DATA_DIR, "210212")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    trial_dirs = load.get_trials_from_fly(fly_dirs)

    for fly_trial_dirs in trial_dirs:
        ref_stack = os.path.join(fly_trial_dirs[0], load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
        ref_offset_dir = os.path.join(fly_trial_dirs[0], load.PROCESSED_FOLDER, "com_offset.npy")
        
        for trial_dir in fly_trial_dirs:
            print("processing trial: " + trial_dir)
            processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)

            stack1 = os.path.join(processed_dir, load.RAW_RED_TIFF)
            stack2 = os.path.join(processed_dir, load.RAW_GREEN_TIFF)
            stack1_out_dir = os.path.join(processed_dir, "red_com_warped.tif")
            stack2_out_dir = os.path.join(processed_dir, "green_com_warped.tif")
            w_output = os.path.join(processed_dir, "motion_field_com.npy")

            offset_dir = os.path.join(processed_dir, "com_offset.npy")

            warp(stack1=stack1, stack2=stack2, ref_stack=ref_stack, 
                 stack1_out_dir=stack1_out_dir, stack2_out_dir=stack2_out_dir, 
                 com_pre_reg=True, offset_dir=offset_dir, ref_offset_dir=ref_offset_dir,
                 return_stacks=False, overwrite=False, select_frames=None,
                 parallel=True, verbose=True, w_output=w_output)
