# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import numpy as np
import os.path
from pathlib import Path
from skimage.transform import resize
from copy import deepcopy
import scipy.stats
import types
import subprocess
import signal

from utils2p import load_img, save_img

def get_stack(stack):
    if stack is None:
        return None
    if isinstance(stack, str) and os.path.isfile(stack):
        stack = load_img(stack)
    elif isinstance(stack, str):
        raise FileNotFoundError("Could not find file: " + stack)
    
    if not isinstance(stack, np.ndarray):
        raise NotImplementedError("stack is not a numpy array.")

    return stack

def save_stack(path, stack):
    save_img(path, stack)

def torch_to_numpy(x):
    return x.detach().cpu().data.numpy()

def resize_stack(stack, size=(128, 128)):
    res_stack = np.zeros((stack.shape[0], size[0], size[1]), np.float32)
    for i in range(stack.shape[0]):
        res_stack[i] = resize(stack[i], size)
    return res_stack

def crop_stack(stack, crop):
    if crop is not None and len(crop) == 2:
        assert crop[0]*2 < stack.shape[1]
        assert crop[1]*2 < stack.shape[2]
        stack = stack[:, crop[0]:stack.shape[1]-crop[0], crop[1]:stack.shape[2]-crop[1]]
    elif crop is not None and len(crop) == 4:
        assert crop[0] < stack.shape[1]
        assert crop[1] < stack.shape[1]
        assert crop[2] < stack.shape[2]
        assert crop[3] < stack.shape[2]
        stack = stack[:, crop[0]:crop[1], crop[2]:crop[3]]
    elif crop is None:
        return stack
    else:
        raise NotImplementedError("crop should be either of length 2 or 4, or be None")
    return stack

def crop_img(img, crop):
    if len(img.shape) > 2:
        img = np.squeeze(img)
    assert len(img.shape) == 2
    if crop is not None and len(crop) == 2:
        assert crop[0]*2 < img.shape[0]
        assert crop[1]*2 < img.shape[1]
        img = img[crop[0]:img.shape[0]-crop[0], crop[1]:img.shape[1]-crop[1]]
    elif crop is not None and len(crop) == 4:
        assert crop[0] < img.shape[0]
        assert crop[1] < img.shape[0]
        assert crop[2] < img.shape[1]
        assert crop[3] < img.shape[1]
        img = img[crop[0]:crop[1], crop[2]:crop[3]]
    elif crop is None:
        return img
    else:
        raise NotImplementedError("crop should be either of length 2 or 4, or be None")
    return img

def makedirs_safe(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_file(directory, name, file_type=""):
    """
    This function finds a unique file with a given name in
    in the directory.
    Copied from utils2p _find_file

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob("*" + name))
    if len(file_names) > 1:
        raise RuntimeError(
            f"Could not identify {file_type} file unambiguously. Discovered {len(file_names)} {file_type} files in {directory}."
        )
    elif len(file_names) == 0:
        raise FileNotFoundError(f"No {file_type} file found in {directory}")
    return str(file_names[0])

def readlines_tolist(file, remove_empty=True):
    with open(file) as f:
        out = f.readlines()
    out = [line.strip() for line in out]
    if remove_empty:
        out = [line for line in out if line != ""]
    return out

def list_attr(lst, attr):
    return [getattr(l, attr) for l in lst]
def list_fn(lst, fn):
    return [fn(l) for l in lst]

def list_mthd(lst, mthd):
    out = []
    for l in lst:
        m = getattr(l, mthd)
        out.append(l.m())
    return out

def list_join(lst, strs):
    if not isinstance(strs, list):
        strs = [strs]
    out_lst = deepcopy(lst)
    for s in strs:
        out_lst = [os.path.join(this_dir, s) for this_dir in out_lst]
    return out_lst

def sem(array, axis=None):
    """
    compute the standard error of the mean based on sem = std/sqrt(N)
    """
    array = np.array(array)
    if axis is None:
        N = np.prod(array.shape)
    else:
        if not isinstance(axis, tuple):
            axis = axis,
        N = np.prod([array.shape[i] for i in axis])
    return np.std(array, axis) / np.sqrt(N)

def conf_int(array, axis=None):
    """
    compute the 95% confidence interval of the mean based on Gaussian assumption:
    CI = 1.96 * SEM
    """
    const = scipy.stats.norm.ppf(0.975)
    return const * sem(array, axis=axis)

def run_shell_command(command, allow_ctrl_c=True, suppress_output=False):
    if allow_ctrl_c:
        try:
            if suppress_output:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
            else:
                process = subprocess.Popen(command, shell=True)
            process.communicate()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
    else:
        if suppress_output:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        else:
            process = subprocess.Popen(command, shell=True)
        process.communicate()