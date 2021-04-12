import os
import array
import numpy as np

import utils2p

class FrameFromRaw:
    def __init__(self, path, width, height, n_channels, meta_n_z=1, n_z=2):
        self.path = os.path.expanduser(os.path.expandvars(path))
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.meta_n_z = meta_n_z
        self.n_z = n_z
        
        self.init_read_sizes()
        
    def init_read_sizes(self):
        self.image_size = self.width * self.height
        self.t_size = self.width * self.height * self.n_channels * self.n_z
        
    
    def read_last_frame(self):
        # TODO: in online mode, we don't want to read the fly-backframes, but only the real data
        # right now, this function assumes that readl data + flyback frame are being read at once
        # start = time.process_time()
        with open(self.path, "rb") as f:
            f.seek(-2 * self.t_size, os.SEEK_END)  # 2* because it is uint16
            a = array.array("H")  # "H" correspond to uint16
            a.fromfile(f, self.t_size)
        # print(time.process_time() - start)
        # --> takes in the order of 4ms to open, read, and close file
        return self.reshape_byte_array(a)
    
    def read_nth_frame(self, n_frame):
        with open(self.path, "rb") as f:
            f.seek(n_frame * 2 * self.t_size, os.SEEK_SET)  # 2* because it is uint16
            a = array.array("H")  # "H" corresponds to uint16
            a.fromfile(f, self.t_size)
        return self.reshape_byte_array(a)
            
    def reshape_byte_array(self, a):
        out = ()
        a = np.array(a).reshape((-1, self.image_size))  # e.g. 4 x 353280 for a 480x736 frame
        if self.n_z > 1:
            for c in range(self.n_channels):
                out += (np.squeeze(a[c::self.n_channels, :].reshape(
                                   (self.n_z, self.height, self.width)
                                   )[:self.meta_n_z, :, :]),
                       )
        else:
            for c in range(self.n_channels):
                out += (np.array(a[c * image_size : (c + 1) * image_size]
                                ).reshape((height, width)),
                       )
        return out
                    
class FrameFromRawMetadata(FrameFromRaw):
    def __init__(self, path, metadata, n_z=2):
        super(FrameFromRawMetadata, self).__init__(path=path, 
            width=metadata.get_num_x_pixels(),
            height=metadata.get_num_y_pixels(),
            n_channels=metadata.get_n_channels(),
            meta_n_z=metadata.get_n_z(),
            n_z=n_z)


if __name__ == "__main__":
    USE_METADATA = False

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
    channel_0 = frame[0]
    channel_1 = frame[1]
    print("channel 0 shape: ", channel_0.shape, "\nchannel 1 shape: ", channel_1.shape)
    assert channel_0.shape == (height, width)
    assert channel_1.shape == (height, width)
