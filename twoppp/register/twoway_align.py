"""
sub-module to perfrom two-way alignment of back- and forth
scanned galvo-resonance images.
Functions copied and modified from:
https://github.com/NeLy-EPFL/noisy2way
"""

import os
import numpy as np

from twoppp import load, utils

def find_shift(stack):
    """
    Finds the appropriate two way alignment shift for a stack.
    The stack is separated into even and odd rows. The shift is then
    determined by comparing the l1 norm of the difference between the
    two halfs of the image for different displacements.
    Parameters
    ----------
    stack : 3D numpy array
        Stack of frames that should be corrected.
        First dimension is time. Second and third dimension are
        spatial.
   
    Returns
    -------
    shift : int
        Optimal shift.
    """
    half0 = stack[:, ::2, :]
    half1 = stack[:, 1::2, :]

    assert half0.shape == half1.shape

    max_shift = 20
    l1_errors = np.zeros(2 * max_shift + 1)
    l2_errors = np.zeros(2 * max_shift + 1)
    shift_values = list(range(-max_shift, max_shift + 1))

    for i, shift in enumerate(shift_values):
        diff = np.zeros((half1.shape[0], half1.shape[0] - abs(shift)), dtype=stack.dtype)
        if shift < 0:
            diff = half1[:, :, abs(shift) :] - half0[:, :, :shift]
        elif shift == 0:
            diff = half1 - half0
        else:
            diff = half1[:, :, :-shift] - half0[:, :, shift:]
        l2 = np.sum(diff ** 2)
        l1 = np.sum(np.abs(diff))
        l1_errors[i] = l1
        l2_errors[i] = l2
    return shift_values[np.argmin(l1_errors)]

def apply_shift(stack, shift):
    """
    Applies a shift value to a stack.
    Parameters
    ----------
    stack : 3D numpy array
        Stack of frames that should be corrected by shift.
        First dimension is time. Second and third dimension are
        spatial.
    shift : int
        Shift magnitude.
    Returns
    -------
    stack : 3D numpy array
        Stack of frames after applying shift.
        First dimension is time. Second and third dimension are
        spatial.
    """
    if shift < 0:
        off_set = 1
        shift = abs(shift)
    elif shift > 0:
        off_set = 0
    else:
        return stack
    # print(off_set, shift)
    stack[:, off_set::2, :-shift] = stack[:, off_set::2, shift:]
    return stack


def align_stacks(stack1, stack1_out, stack2=None, stack2_out=None, shift=None,
                 shift_out=None, overwrite=False, return_stacks=False):
    if os.path.isfile(stack1_out) and not overwrite:
        if stack2 is None or os.path.isfile(stack2_out):
            return
    stack1 = utils.get_stack(stack1)
    stack2 = utils.get_stack(stack2)

    if os.path.isfile(shift_out):
        shift = np.load(shift_out)
    shift = find_shift(stack1) if shift is None else shift

    stack1_shifted = apply_shift(stack1, shift)
    stack2_shifted = apply_shift(stack2, shift) if stack2 is not None else None

    if stack1_out is not None:
        utils.save_stack(stack1_out, stack1_shifted)
    if stack2_shifted is not None and stack2_out is not None:
        utils.save_stack(stack2_out, stack2_shifted)
    if shift_out is not None:
        np.save(shift_out, shift)
    if return_stacks:
        return stack1_shifted, stack2_shifted