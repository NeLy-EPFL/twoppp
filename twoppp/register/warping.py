"""
sub-module to perfrom center of mass registration 
and non-affine motion correction using the ofco package.
"""

import os
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
import cv2

import utils2p
import ofco

from twoppp import load, utils

def save_ref_frame(stack, ref_frame_dir, i_frame = 0, com_pre_reg=True, overwrite=False, crop=None):
    """get reference frame for motion correction.
    If requested, compute center of mass registration and crop

    Parameters
    ----------
    stack : numpy array or str
        image stack where reference frame is taken from

    ref_frame_dir : str
        output path

    i_frame : int, optional
        frame index, by default 0

    com_pre_reg : bool, optional
        whether to perform center of mass registration on reference frame, by default True

    overwrite : bool, optional
        whether to overwrite existing results, by default False

    crop : list, optional
        list of length 2 for symmetric cropping (same on both sides),
        or list of length 4 for assymetric cropping, by default None
    """
    if os.path.isfile(ref_frame_dir) and not overwrite:
        return
    stack = utils.get_stack(stack)
    ref_frame = stack[i_frame:i_frame+1, :, :]
    if com_pre_reg:
        ref_frame = center_stack(ref_frame, return_offset=False)
    ref_frame = utils.crop_img(ref_frame, crop)
    utils2p.save_img(ref_frame_dir, ref_frame)

def center_and_crop(stack1, stack2=None, crop=None, stack1_out_dir=None, stack2_out_dir=None,
                    offset_dir=None, overwrite=False, return_stacks=False):
    """compute and apply center of mass (COM) registration on entire stack and crop.
    Can also be used with pre-computed shifts. Supply them in offset_dir.

    Parameters
    ----------
    stack1 : numpy array or str
        reference stack of images, based on which the COM offsets will be computed

    stack2 : numpy array or str, optional
        stack to which the offset will also be applied, by default None

    crop : list, optional
        list of length 2 for symmetric cropping (same on both sides),
        or list of length 4 for assymetric cropping, by default None

    stack1_out_dir : str, optional
        where to save stack1 after COM and crop, by default None

    stack2_out_dir : str, optional
         where to save stack2 after COM and crop, by default None

    offset_dir : str, optional
        where to save the COM offsets. can also be used to load pre-computed shifts, by default None

    overwrite : bool, optional
        whether to overwrite existing results.
        Currently overwrite only possible on final stack, 
        not if offset_dir already is a file, by default False

    return_stacks : bool, optional
        whether to return stacks, by default False

    Returns
    -------
    (stack1_shifted: numpy array)
        only returns if return_stacks == True. Otherwise returns None

    (stack2_shifted: numpy array)
        only returns if return_stacks == True. Otherwise returns None
    """
    if os.path.isfile(stack1_out_dir) and not overwrite:
        if not return_stacks:
            return None, None
        stack1_shifted = utils2p.load_img(stack1_out_dir)
        if stack2 is not None and os.path.isfile(stack2_out_dir):
            stack2_shifted = utils2p.load_img(stack2_out_dir)
        else:
            stack2_shifted = None
        return stack1_shifted, stack2_shifted

    stack1 = utils.get_stack(stack1)
    stack2 = utils.get_stack(stack2)

    if os.path.isfile(offset_dir):
        offsets = np.load(offset_dir)
        stack1_shifted = apply_offset(stack1, offsets)
    else:
        stack1_shifted, offsets = center_stack(stack1, return_offset=True)
    stack2_shifted = apply_offset(stack2, offsets) if stack2 is not None else None

    stack1_shifted = utils.crop_stack(stack1_shifted, crop)
    stack2_shifted = utils.crop_stack(stack2_shifted, crop) if stack2 is not None else None

    if offset_dir is not None:
        np.save(offset_dir, offsets)
    if stack1_out_dir is not None:
        utils2p.save_img(stack1_out_dir, stack1_shifted)
    if stack2_out_dir is not None and stack2 is not None:
        utils2p.save_img(stack2_out_dir, stack2_shifted)

    return stack1_shifted, stack2_shifted if return_stacks else None, None

def center_stack(frames, return_offset=False, sigma_filt=(0, 10, 10), foreground_thres=0.75):
    """perform center of mass registration on a stack of images.
    Smoothes and thresholds the stack before computing offset
    (return stack is not smoothed and thresholded).
    can return offsets to apply to other (parallely acquired) stack

    Parameters
    ----------
    frames : numpy array
        stack of frames to be centered

    return_offset : bool, optional
        whether to return the offset, by default False

    sigma_filt : tuple, optional
        smoothing kernel to be applied to the stack before performing COM registration,
        by default (0, 10, 10)

    foreground_thres : float, optional
        quantile threshold applied to the stack before computing shift, by default 0.75

    Returns
    -------
    frames_shifted: numpy array
        COM registered frames

    (offsets: numpy array)
        only if return_offset == True
    """
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
    """apply shifts to a stack of frames, e.g. for center of mass registration.

    Parameters
    ----------
    frames : numpy array
        stack of frames

    offsets : numpy array
        array of size N_frames, 2 containing the shifts to be applied

    Returns
    -------
    frames_shifted: numpy array
    """
    if frames.ndim == 2:
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
         select_frames=None, parallel=True, verbose=False, w_output=None, initial_w=None, 
         save_motion_field=True, param=None, crop_roi=False):
    """use optic flow motion correction (ofco) to perform non-affine registraion.
    Wrapper around https://github.com/NeLy-EPFL/ofco
    Will try to use existing w_output and apply it to stack2.

    Parameters
    ----------
    stack1 : numpy array or str
        reference stack

    stack2 : numpy array or str, optional
        2nd stack to which motion field will be applied, by default None

    ref_frame : int, numpy array or str, optional
        reference frame to register to
        if int, will be interpreted as frame index.
        if otherwise, will take array or load .tif from file, by default None

    stack1_out_dir : str, optional
        where to save stack1 after registration, by default None

    stack2_out_dir : [type], optional
        where to save stack2 after refistration, by default None

    com_pre_reg : bool, optional
        whether to perform center of mass registration before ofco, by default False

    offset_dir : str, optional
        COM registration offsets. if specified, will load them.
        if not a file, will save there, by default None

    return_stacks : bool, optional
        whether to return the stacks, by default False

    overwrite : bool, optional
        whether to overwrite existing results, by default False

    select_frames : list, optional
        if specified, only perform motion correction on a selected range of frames, by default None

    parallel : bool, optional
        whether to use multiprocessing.pool to parralelise, by default True

    verbose : bool, optional
        whether to print intermediate steps, by default False

    w_output : str, optional
        location to store the ofco motion fields, by default None

    initial_w : numpy array, optional
        weights to prime the optimisation, by default None

    save_motion_field : bool, optional
        whether to save the motion fields to file, by default True

    param : dict, optional
        parameters for ofco. if not specified, use ofco internal defaults, by default None

    Returns
    -------
    (stack1_warped: numpy array)
        only if return_stacks is selected

    (stack2_warped: numpy array)
        only if return_stacks is selected

    Raises
    ------
    NotImplementedError
        if stack1 is not a numpy array or cannot be found
    """
    
    param = ofco.utils.default_parameters() if param is None else param

    if os.path.isfile(stack1_out_dir) and not overwrite:
        if stack2 is not None and stack2_out_dir is not None and w_output is not None \
            and not os.path.isfile(stack2_out_dir):
            try:
                print("Applying motion field to stack2")
                apply_warp(stack2, stack2_out_dir, w_output)
            except:
                print("Stack1 is already warped, stack2 is not. \n"+\
                      "Could not find motion weights. If you want to recalculate, " +\
                          "select 'overwrite' flag.")
        if not return_stacks and not crop_roi:
            return None, None

        if not return_stacks and crop_roi:
            print("Finding ROI")
            stack1_warped = utils2p.load_img(stack1_out_dir)
            coords = find_region_to_crop(stack1_warped)
            
            print("Saving original warped stacks")
            stack2_warped = utils2p.load_img(stack2_out_dir)
            utils2p.save_img(stack1_out_dir.replace("warped","warped_noCrop"), stack1_warped)
            utils2p.save_img(stack2_out_dir.replace("warped","warped_noCrop"), stack2_warped)
            
            print("Cropping warped stacks")
            stack1_warped = utils.crop_stack(stack1_warped, coords)
            stack2_warped = utils.crop_stack(stack2_warped, coords)
            utils2p.save_img(stack1_out_dir, stack1_warped)
            utils2p.save_img(stack2_out_dir, stack2_warped)
            
            return None, None

        print("Loading warped stacks")
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
        raise NotImplementedError("stack1 has to be a numpy array")

    N_frames, N_y, N_x = stack1.shape

    if stack2 is not None:
        stack2 = utils.get_stack(stack2)
        assert stack1.shape == stack2.shape
        if com_pre_reg:
            stack2 = apply_offset(stack2, offsets)

    if isinstance(ref_frame, int):
        assert ref_frame < N_frames
    elif ref_frame is not None:
        ref_frame = utils.get_stack(ref_frame)
        ref_frame = np.squeeze(ref_frame)
        if ref_frame.ndim == 3:
            ref_frame = ref_frame[0, :, :]
        assert ref_frame.shape == (N_y, N_x)
    else:
        ref_frame = 0

    stack1_warped, stack2_warped = ofco.motion_compensate(
        stack1, stack2,
        frames=range(N_frames) if select_frames is None else select_frames,
        param=param, verbose=verbose, parallel=parallel,
        w_output=w_output if save_motion_field else None,
        initial_w=initial_w, ref_frame=ref_frame)
        
    if stack1_out_dir is not None:
        utils2p.save_img(stack1_out_dir, stack1_warped)
    if stack2_out_dir is not None:
        utils2p.save_img(stack2_out_dir, stack2_warped)

    return stack1_warped, stack2_warped if return_stacks else None, None

def find_region_to_crop(stack):
    mean_img = np.zeros_like(stack[0, :, :], dtype=np.int64)
    for img in stack:
        mean_img += img
    mean_img = mean_img/stack.shape[0]
    mean_img = mean_img.astype(np.uint8)
    
    hist = cv2.calcHist([mean_img],[0],None,[256],[0,256])
    vals_hist = hist.T[0]
    max_hist = np.max(vals_hist)
    min_hist = np.min(vals_hist)

    #import matplotlib.pyplot as plt
    #plt.plot(hist)
    #plt.xlim([0,256])
    #plt.savefig("/home/nely/Desktop/his.png")
    
    starting_peak = np.where(vals_hist[10:]>0.1*(max_hist-min_hist))[0]
    threshold = starting_peak[0]
    _,roi = cv2.threshold(mean_img,threshold,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    
    output = cv2.connectedComponentsWithStats(roi_clean, 4, cv2.CV_32S)
    stats = np.transpose(output[2])
    sizes = stats[4]
    label_roi = np.argmax(sizes[1:])+1
    
    left = stats[0][label_roi]
    top = stats[1][label_roi]
    width = stats[2][label_roi]
    height = stats[3][label_roi]
    
    h, w = roi_clean.shape
    #label_img = np.zeros((h,w),np.uint8)
    #label_img[np.where(output[1] == label_roi)] = 255
    
    #cv2.imshow("img",roi)
    #cv2.imshow("img_clean",roi_clean)
    #cv2.imshow("label",label_img)
    #cv2.waitKey()

    gap = 30
    y0 = top - gap if top-gap>0 else 0 
    y1 = top + height + gap if top+height+gap<h else h-1 
    x0 = left - gap if left-gap>0 else 0
    x1 = left + width + gap if left+width+gap<w else w-1

    if (y1-y0)%2>0:
        y0+=1
    if (x1-x0)%2>0:
        x0+=1
    
    return [y0, y1, x0, x1]

def warp_N_parts(stack1, stack1_out_dir, N_parts, stack2=None, stack2_out_dir=None, ref_frame=None,
         com_pre_reg=False, offset_dir=None, return_stacks=False, overwrite=False,
         select_frames=None, parallel=True, verbose=False, w_output=None,initial_w=None,param=None):
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
        raise NotImplementedError("stack1 has to be a numpy array")

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
    """apply motion field to stack. wraps around apply_motion_field() with data loading/saving.

    Parameters
    ----------
    stack : numpy array or str
        stack to which motion field is applied

    stack_out_dir : str
        where to save the stack

    w_output : str
        location of .npy weights

    select_frames : list, optional
        whether to warp only selected frames,
        must be consistent with the time the motion field was computed
        by default None

    com_pre_reg : bool, optional
        whether to apply center of mass registration before, by default False

    offset_dir : str, optional
        COM registration offsets. if specified, will load them.
        if not a file, will save there, by default None

    return_stacks : bool, optional
        whether to return the stacks, by default False

    overwrite : bool, optional
        whether to overwrite existing results, by default False

    Returns
    -------
    (stack_warped: numpy array)
        only returns if return_stacks == True

    Raises
    ------
    NotImplementedError
        if stack is not a numpy array or cannot be found
    """
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
        raise NotImplementedError("stack has to be a numpy array")

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
    """apply motion field using bilinearly interpolation.

    Parameters
    ----------
    stack : numpy array
        stack of frames to apply motion field to

    motion_field : numpy array
        motion field computed using ofco

    Returns
    -------
    stack_warped: numpy array
    """
    N_frames, N_y, N_x = stack.shape
    assert motion_field.shape == (N_frames, N_y, N_x, 2)

    stack_warped = np.zeros_like(stack)

    for i_frame in range(N_frames):
        stack_warped[i_frame, :, :] = ofco.warping.bilinear_interpolate(
            stack[i_frame, :, :],
            motion_field[i_frame, :, :, 0],
            motion_field[i_frame, :, :, 1])

    return stack_warped
