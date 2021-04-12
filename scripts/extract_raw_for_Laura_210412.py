import os, sys
import numpy as np

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
OUT_PATH = os.path.join(MODULE_PATH, "outputs")


from longterm.utils.raw_files import FrameFromRawMetadata
from longterm.load import NAS_DIR

import utils2p

data_dir = os.path.join(NAS_DIR, "LH", "210408", "J1M5_fly11", "cs_caff_after")
raw_dir = utils2p.find_raw_file(data_dir)
metadata_dir = utils2p.find_metadata_file(data_dir)
metadata = utils2p.Metadata(metadata_dir)
myFrameFromRaw = FrameFromRawMetadata(raw_dir, metadata, n_z=2)

frames = [myFrameFromRaw.read_nth_frame(n) for n in range(195)]
green = np.array([frame[0] for frame in frames])
red = np.array([frame[1] for frame in frames])

utils2p.save_img(os.path.join(data_dir, "green.tif"), green)
utils2p.save_img(os.path.join(data_dir, "red.tif"), red)



