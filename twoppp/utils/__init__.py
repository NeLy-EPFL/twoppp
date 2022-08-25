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
import pandas as pd

from utils2p import load_img, save_img

def get_df(df):
    if isinstance(df, str) and os.path.isfile(df):
        df = pd.read_pickle(df)
    if df is not None:
        assert isinstance (df, pd.DataFrame)
    return df

def get_stack(stack):
    """load a .tif image from file
    Wrapper around utils2p.load_img() with added flexibility

    Parameters
    ----------
    stack : str or numpy array or None
        absolute path of stack
        if already a numpy array, just pass through
        if None, return None

    Returns
    -------
    stack: numpy array or None

    Raises
    ------
    FileNotFoundError
        if str is not a file

    NotImplementedError
        if after loading from file stack is not a numpy array
    """
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
    """Wrapper around utils2p.save_img()

    Parameters
    ----------
    path : str
        location to save to. should be .tif

    stack : numpy array
        stack to save
    """
    save_img(path, stack)

def torch_to_numpy(x):
    return x.detach().cpu().data.numpy()

def resize_stack(stack, size=(128, 128)):
    """interpolate a stack of images to a different size using 
    resize from skimage.transform

    Parameters
    ----------
    stack : numpy array
        stack of images

    size : tuple, optional
        new size, by default (128, 128)

    Returns
    -------
    res_stack: numpy array
        resized stack
    """
    res_stack = np.zeros((stack.shape[0], size[0], size[1]), np.float32)
    for i in range(stack.shape[0]):
        res_stack[i] = resize(stack[i], size)
    return res_stack

def crop_stack(stack, crop):
    """crop stack

    Parameters
    ----------
    stack : numpy array
        stack of images to be cropped

    crop : list or tuple
        list of length 2 for symmetric cropping (crop specified value on both sides),
            stack[:, crop[0]:stack.shape[1]-crop[0], crop[1]:stack.shape[2]-crop[1]]
        or list of length 4 for assymetric cropping, by default None:
            stack[:, crop[0]:crop[1], crop[2]:crop[3]]

    Returns
    -------
    stack: numpy array
        cropped stack

    Raises
    ------
    NotImplementedError
        if the list/tuple is not of length 2 or 4
    """
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
    """crop a single image.

    Parameters
    ----------
    img : numpy array
        image

    crop : list or tuple
         list of length 2 for symmetric cropping (crop specified value on both sides),
            img[crop[0]:stack.shape[1]-crop[0], crop[1]:stack.shape[2]-crop[1]]
        or list of length 4 for assymetric cropping, by default None:
            img[crop[0]:crop[1], crop[2]:crop[3]]


    Returns
    -------
    img: numpy array
        cropped image

    Raises
    ------
    NotImplementedError
        if the list/tuple is not of length 2 or 4
    """
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
    """wrapper around os.makedirs().
    Only makes directory if it does not already exist

    Parameters
    ----------
    path : str
        directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def find_file(directory, name, file_type="", raise_error=True):
    """
    This function finds a unique file with a given name in
    in the directory.
    Copied and modified from utils2p _find_file:
    https://github.com/NeLy-EPFL/utils2p

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
        if raise_error:
            raise RuntimeError(
                f"Could not identify {file_type} file unambiguously." + \
                f"Discovered {len(file_names)} {file_type} files in {directory}."
            )
        else:
            return file_names
    elif len(file_names) == 0:
        if raise_error:
            raise FileNotFoundError(f"No {file_type} file found in {directory}")
        else:
            return None
    return str(file_names[0])

def readlines_tolist(file, remove_empty=True):
    """read the lines of a .txt file into a list

    Parameters
    ----------
    file : str
        location of .txt file

    remove_empty : bool, optional
        remove empty entries, by default True

    Returns
    -------
    list
        list of strings
    """
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
    """compute the standard error of the mean based on sem = std/sqrt(N)

    Parameters
    ----------
    array : numpy array

    axis : int, optional
        along which axis to compute the sem, by default None

    Returns
    -------
    numpy array
        standard error of the mean along specified axis
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
    """compute the 95% confidence interval of the mean based on Gaussian assumption:
    CI = 1.96 * SEM

    Parameters
    ----------
    array : numpy array

    axis : int, optional
        along which axis to compute the sem, by default None

    Returns
    -------
    numpy array
        confidence interval along specified axis
    """
    const = scipy.stats.norm.ppf(0.975)
    return const * sem(array, axis=axis)

def run_shell_command(command, allow_ctrl_c=True, suppress_output=False):
    """use the subprocess module to run a shell command

    Parameters
    ----------
    command : str
        shell command to execute

    allow_ctrl_c : bool, optional
        whether a CTRL+C event will allow to continue or not, by default True

    suppress_output : bool, optional
        whether to not show outputs, by default False
    """
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

def standardise(array, axis=0):
    repeats = array.shape[axis] if isinstance(axis, int) else [array.shape[ax] for ax in axis]
    mu = np.repeat(np.mean(array, axis=axis, keepdims=True), repeats=repeats, axis=axis)
    std = np.repeat(np.std(array, axis=axis, keepdims=True), repeats=repeats, axis=axis)
    return (array - mu) / std

def zscore(array, axis=0):
    return standardise(array, axis=axis)

def normalise(array, arange=[0, 1], axis=0):
    repeats = array.shape[axis] if isinstance(axis, int) else [array.shape[ax] for ax in axis]
    amin = np.repeat(np.min(array, axis=axis, keepdims=True), repeats=repeats, axis=axis)
    amax = np.repeat(np.max(array, axis=axis, keepdims=True), repeats=repeats, axis=axis)
    array_norm = (array - amin) / (amax - amin)
    return array_norm * (arange[1] - arange[0]) + arange[0]

def normalise_quantile(array, q=0.99, arange=[0, 1], axis=0):
    if axis is None:
        if isinstance(q, list) or isinstance(q, tuple):
            qmin = np.quantile(array, q=q[0], axis=axis, keepdims=True)
            qmax = np.quantile(array, q=q[1], axis=axis, keepdims=True)
        else:
            qmin = np.quantile(array, q=0.5*(1-q), axis=axis, keepdims=True)
            qmax = np.quantile(array, q=0.5*(1+q), axis=axis, keepdims=True)
    elif isinstance(axis, int):
        repeats = array.shape[axis]
        if isinstance(q, list) or isinstance(q, tuple):
            qmin = np.repeat(np.quantile(array, q=q[0], axis=axis, keepdims=True), repeats=repeats, axis=axis)
            qmax = np.repeat(np.quantile(array, q=q[1], axis=axis, keepdims=True), repeats=repeats, axis=axis)
        else:
            qmin = np.repeat(np.quantile(array, q=0.5*(1-q), axis=axis, keepdims=True), repeats=repeats, axis=axis)
            qmax = np.repeat(np.quantile(array, q=0.5*(1+q), axis=axis, keepdims=True), repeats=repeats, axis=axis)
    else:
        raise NotImplementedError(f"axis mus be either None or of type int, but it was {type(axis)}")
    array_norm = (array - qmin) / (qmax - qmin)
    return array_norm * (arange[1] - arange[0]) + arange[0]
