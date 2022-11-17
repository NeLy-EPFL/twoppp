#!/usr/bin/env python3

# largely copied and slightly modified from
# https://github.com/NeLy-EPFL/ofco/blob/master/examples/register.py

import sys
import os.path
import glob
import tempfile
import shutil
from datetime import datetime

from skimage import io
import numpy as np

import utils2p

from ofco import motion_compensate
from ofco.utils import default_parameters


print(datetime.now().strftime('Started at %H:%M:%S on %A %h %d, %Y'))

STACK_IN = "red_com_crop.tif"
STACK_OUT = "red_com_warped.tif"

def reassamble_warped_images(folder):
    n_files = len(glob.glob(os.path.join(folder, STACK_OUT[:-4]+"*.tif")))
    stacks = []
    for i in range(n_files):
        substack = utils2p.load_img(os.path.join(folder, STACK_OUT[:-4]+f"_{i}.tif"))
        stacks.append(substack)
    stack = np.concatenate(stacks, axis=0)
    utils2p.save_img(os.path.join(folder, STACK_OUT), stack)

def reassamble_vector_fields(folder):
    n_files = len(glob.glob(os.path.join(folder, f"w_*.npy")))
    vector_fields = []
    for i in range(n_files):
        path = os.path.join(folder, f"w_{i}.npy")
        sub_fields = np.load(path)
        vector_fields.append(sub_fields)
    vector_field = np.concatenate(vector_fields, axis=0)
    np.save(os.path.join(folder, f"w.npy"), vector_field)

def copy_read_delete_img(stack_file):
    tmp = tempfile.NamedTemporaryFile(delete=True)
    shutil.copy2(stack_file, tmp.name)
    stack = io.imread(tmp.name)
    if os.path.isfile(tmp.name):
        os.remove(tmp.name)
    return stack

#folder = "/scratch/aymanns/200901_G23xU1/Fly1/001_coronal"
#ref_frame = io.imread("/scratch/aymanns/200901_G23xU1/Fly1/ref_frame.tif")
folder = sys.argv[1]
ref_frame = sys.argv[2]
print("Folder:", folder)
print("Reference frame:", ref_frame)
# ref_frame = io.imread(ref_frame)
ref_frame = copy_read_delete_img(ref_frame)

param = default_parameters()
#param["lmbd"] = 4000
for i, substack in enumerate(utils2p.load_stack_batches(os.path.join(folder, STACK_IN), 28)):
    print(i)
    frames = range(len(substack))
    warped_output = os.path.join(folder, STACK_OUT[:-4]+f"_{i}.tif")
    w_output = os.path.join(folder, f"w_{i}.npy")
    if os.path.isfile(warped_output) and os.path.isfile(w_output):
        print("skipped because it exists")
        continue
    stack1_warped, stack2_warped = motion_compensate(
        substack, None, frames, param, parallel=True, verbose=True,
        w_output=w_output, ref_frame=ref_frame
    )

    io.imsave(warped_output, stack1_warped)
#io.imsave(os.path.join(folder, "warped2.tif"), stack2_warped)

reassamble_warped_images(folder)
reassamble_vector_fields(folder)

print(datetime.now().strftime('Finished at %H:%M:%S on %A %h %d, %Y'))
