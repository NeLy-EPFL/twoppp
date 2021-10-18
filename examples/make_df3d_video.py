from twoppp.plot import videos

if __name__ == "__main__":
    # this should only be run if pose estimation with deepfly3d has been run before.
    trial_dir = "path/to/trial"
    out_dir = "path/trial/processed"
    videos.make_video_df3d(
        trial_dir,
        out_dir,
        print_frame_num=False,
        print_frame_time=True,
        print_beh_label=False,  # could be used to write the behavioural label from classification
        beh_label_dir=None,
        video_name="df3d.mp4",
        cameras=[5],  # which camera to use. could also be multiple
        downsample=1,
        speedup=1)