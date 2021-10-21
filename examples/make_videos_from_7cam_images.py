import os.path

from twoppp import load
from twoppp.plot.videos import make_video_from_cam_frames

if __name__ == "__main__":
    fly_dir = os.path.join(load.NAS2_DIR_JB, "211005_J1M5", "Fly2")
    trial_dir = os.path.join(fly_dir, "001_xz")
    images_dir = os.path.join(trial_dir, "behData", "images")
    N_cams = 7
    for cam in range(N_cams):
        make_video_from_cam_frames(images_dir, cam,
                                    required_n_frames=None,
                                    video_name="_test_camera")
