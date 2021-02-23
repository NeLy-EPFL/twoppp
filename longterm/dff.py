# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import numpy as np
from scipy.ndimage.filters import gaussian_filter

import nely_suite

from longterm.utils import get_stack

def compute_dff_from_stack(stack, baseline_blur=3, baseline_mode="convolve", # slow alternative: "quantile"
                           baseline_length=10, basline_quantile=0.05,
                           use_crop=True, manual_add_to_crop=20,
                           dff_blur=0, dff_out_dir=None, return_stack=True):
    # load from path in case stack is a path. if numpy array, then just continue
    stack = get_stack(stack)
    N_frames, N_x, N_y = stack.shape

    # 1. blur stack if required
    stack_blurred = gaussian_filter(stack, (0, baseline_blur, baseline_blur)) if baseline_blur else stack
    
    # 2. compute baseline
    if baseline_mode == "convolve":
        dff_baseline = nely_suite.find_pixel_wise_baseline(stack_blurred, n=baseline_length)
    elif baseline_mode == "quantile":
        dff_baseline = nely_suite.quantile_baseline(stack_blurred, basline_quantile)
    elif isinstance(baseline_mode, np.ndarray) and baseline_mode.shape == (N_x, N_y):
        dff_baseline = baseline_mode
    else:
        raise(NotImplementedError)
        
    # 3. compute cropping indices
    if use_crop:
        mask = nely_suite.analysis.background_mask(stack_blurred, z_projection="std", threshold="otsu")
        idx = np.where(mask)
        y_min = np.maximum(np.min(idx[0]) - manual_add_to_crop, 0)
        y_max = np.minimum(np.max(idx[0]) + manual_add_to_crop, stack_blurred.shape[1])
        x_min = np.maximum(np.min(idx[1]) - manual_add_to_crop, 0)
        x_max = np.minimum(np.max(idx[1]) + manual_add_to_crop, stack_blurred.shape[2])

        #4. apply cropping
        stack = nely_suite.crop_stack(stack, x_min, x_max, y_min, y_max)
        stack_blurred = nely_suite.crop_stack(stack_blurred, x_min, x_max, y_min, y_max)
        dff_baseline = dff_baseline[y_min : y_max + 1, x_min : x_max + 1]
        
    # 5. compute dff
    # this also applies a median filter with (3,3,3) kernel 
    # and ignores the areas set to 0 by motion correction
    dff = nely_suite.calculate_dff(stack,
                                   dff_baseline, 
                                   apply_filter=True, occlusion_handling=True)
    
    # 6. post-process dff
    dff = gaussian_filter(dff, (0, dff_blur, dff_blur)) if dff_blur else dff

    if dff_out_dir is not None:
        nely_suite.save_img(dff_out_dir, dff)

    if return_stack:
        return dff
    else:
        return None