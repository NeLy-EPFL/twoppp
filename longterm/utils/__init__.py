# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import numpy as np
import os.path

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