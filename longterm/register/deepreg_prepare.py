# Jonas Braun
# jonas.braun@epfl.ch
# 19.02.2021

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import zscore
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
from copy import deepcopy
import cv2

import utils2p

FILE_PATH = os.path.realpath(__file__)
REGISTER_PATH, _ = os.path.split(FILE_PATH)
LONGTERM_PATH, _ = os.path.split(REGISTER_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils
from longterm.plot.videos import make_video_2p, make_multiple_video_2p
from longterm.register.warping import apply_offset, center_stack

def resize_stack(stack, size=(128, 128)):
    res_stack = np.zeros((stack.shape[0], size[0], size[1]), np.float32)
    for i in range(stack.shape[0]):
        res_stack[i] = resize(stack[i], size)
    return res_stack

def crop_stack(stack, crop_size):
    return stack[:, crop_size[0]:-crop_size[0], crop_size[1]:-crop_size[1]]

def prepare_data(stacks, data_out_dir, resize=(128, 128), center=True,
                 center_out_dirs=None, crop=True, crop_out_dir=None, crop_size=None):
    if not isinstance(stacks, list):
        stacks = [stacks]
    stacks = [utils.get_stack(stack).astype(np.float32) for stack in stacks]
    
    # center each frame
    if center:
        centered = []
        offsets = []
        for i_stack, (stack, center_out_dir) in enumerate(zip(stacks, center_out_dirs)):
            if os.path.isfile(center_out_dir):
                offset = np.load(center_out_dir)
                stack = apply_offset(stack, offset)
            else:
                stack, offset = center_stack(stack, return_offset=True)
                if center_out_dir is not None: #TODO:
                    np.save(center_out_dir, offset)

            centered.append(stack)
            offsets.append(offset)
            
        stacks = centered

    if crop:
        # pixelmax = np.max(np.concatenate([np.max(stack, axis=0, keepdims=True) for stack in stacks]), axis=0)
        if crop_size is None:
            crop_size = np.abs(np.mean(offsets, axis=(0, 1))).astype(np.int)

        stacks = [crop_stack(stack, crop_size) for stack in stacks]

    # resize each stack
    if resize == "max":
        resize = np.min(stacks[0].shape[1:2])
        resize = (resize, resize)
    if resize is not None:
        stacks = [resize_stack(stack, size=resize) for stack in stacks]

    # zscore each stack
    stacks = [(stack - stack.mean()) / stack.std() for stack in stacks]

    # concatenate stacks
    all_stacks = np.concatenate(stacks, axis=0)

    np.save(data_out_dir, all_stacks)


if __name__ == "__main__":
    PREP_DATA = True
    TEST_COM = False
    CHECK_DATA = False
    REDUCE_DATA = False

    if PREP_DATA:
        date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
        fly_dirs = load.get_flies_from_datedir(date_dir)
        trial_dirs = load.get_trials_from_fly(fly_dirs)

        stacks = [os.path.join(trial_dir, load.PROCESSED_FOLDER, load.RAW_RED_TIFF) for trial_dir in trial_dirs[0]]
        
        data_out_dir = os.path.join(fly_dirs[0], load.PROCESSED_FOLDER)
        if not os.path.isdir(data_out_dir):
            os.makedirs(data_out_dir)
        data_out_dir_128 = os.path.join(data_out_dir, "reg_train_data_128.npy")
        data_out_dir_256 = os.path.join(data_out_dir, "reg_train_data_256.npy")
        data_out_dir_max = os.path.join(data_out_dir, "reg_train_data_max.npy")
        data_out_dir_full = os.path.join(data_out_dir, "reg_train_data_full.npy")


        center_out_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER, "com_offset.npy") for trial_dir in trial_dirs[0]]
        crop_out_dir = os.path.join(data_out_dir, "reg_train_crop.npy")
        """
        prepare_data(stacks=stacks, data_out_dir=data_out_dir_128, resize=(128,128), 
                     center=True, center_out_dirs=center_out_dirs,
                     crop=True, crop_out_dir=crop_out_dir)
        """
        prepare_data(stacks=stacks, data_out_dir=data_out_dir_256, resize=(256,256), 
                     center=True, center_out_dirs=center_out_dirs,
                     crop=True, crop_out_dir=crop_out_dir, crop_size=(32,48))
        """
        prepare_data(stacks=stacks, data_out_dir=data_out_dir_max, resize="max", 
                     center=True, center_out_dirs=center_out_dirs,
                     crop=True, crop_out_dir=crop_out_dir)
        prepare_data(stacks=stacks, data_out_dir=data_out_dir_full, resize=None, 
                     center=True, center_out_dirs=center_out_dirs,
                     crop=True, crop_out_dir=crop_out_dir)
        """

    if TEST_COM: 
        red = utils2p.load_img("/home/jbraun/data/longterm/210212/Fly1/cs_003/processed/red.tif")
        frames = np.arange(100)
        red = red[frames, :, :]
        """
        # this is a very basic test to try out Center of mass registration with a moving dot
        red = np.zeros((10, 500, 500))
        x = np.linspace(start=130, stop=370, num=10).astype(np.int)
        y = np.linspace(start=130, stop=370, num=10).astype(np.int)
        for i in range(10):
            red[i, x[i], y[i]] = 50
        """
        red_com_reg, offsets = center_stack(red, return_offset=True)  # center_stack(red_filt)

        make_multiple_video_2p(greens=[np.zeros_like(red) for _ in range(2)], out_dir="/home/jbraun/data/longterm/210212/Fly1/processed/", video_name="com_test",
                            reds=[red, red_com_reg], frame_rate=10)


    if CHECK_DATA:
        data = np.load("/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_128.npy")
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.imshow(data[0])
        fig.savefig("/home/jbraun/data/longterm/210212/Fly1/processed/test_data.png")

    if REDUCE_DATA:
        data = np.load("/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_full.npy")
        data2 = data[:, 3:-3, 5:-5]
        np.save("/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_640_416.npy", data2)
        np.random.seed(123)
        randinds = np.random.randint(low=0, high=data.shape[0], size=data.shape[0]// 4)
        data2_red = data2[randinds, :, :]
        np.save("/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_640_416_red.npy", data2_red)

    pass
