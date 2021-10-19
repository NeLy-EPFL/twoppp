import os.path

import utils2p

from twoppp.utils.raw_files import FrameFromRawMetadata, FrameFromRaw

if __name__ == "__main__":
    USE_METADATA = True

    data_dir = "/home/jbraun/data/210216_J1xCI9/Fly1/001_xz/2p"
    raw_dir = utils2p.find_raw_file(data_dir)

    if USE_METADATA:
        metadata_dir = utils2p.find_metadata_file(data_dir)
        metadata = utils2p.Metadata(metadata_dir)
        myFrameFromRaw = FrameFromRawMetadata(raw_dir, metadata, n_z=2)
    else:
        height, width = 480, 736
        n_channels = 2
        myFrameFromRaw = FrameFromRaw(raw_dir, width=width, height=height,
                                      n_channels=n_channels,
                                      meta_n_z=1, n_z=2)

    frame = myFrameFromRaw.read_nth_frame(100)
    green = frame[0]
    red = frame[1]
    print("green channel shape: ", green.shape, "\nred channel shape: ", red.shape)
    assert green.shape == (height, width)
    assert green.shape == (height, width)

    utils2p.save_img(os.path.join(data_dir, "green.tif"), green)
    utils2p.save_img(os.path.join(data_dir, "red.tif"), red)
