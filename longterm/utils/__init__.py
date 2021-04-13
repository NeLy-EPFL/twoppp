# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import numpy as np
import os.path
from pathlib import Path

from utils2p import load_img

def get_stack(stack):
    if isinstance(stack, str) and os.path.isfile(stack):
        stack = load_img(stack)
    elif isinstance(stack, str):
        raise FileNotFoundError
    
    if not isinstance(stack, np.ndarray):
        raise NotImplementedError

    return stack

def torch_to_numpy(x):
        return x.detach().cpu().data.numpy()

def makedirs_safe(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def find_file(directory, name, file_type):
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