from twoppp.plot import videos

if __name__ == "__main__":
    trial_dirs = [
        "path/to/fly/trial1",
        "path/to/fly/trial2"
    ]
    greens = [
        "path/to/fly/trial1/processed/green.tif",
        "path/to/fly/trial2/processed/green.tif"
        ]
    reds = [
        "path/to/fly/trial1/processed/red.tif",
        "path/to/fly/trial2/processed/red.tif"
        ]
    dffs = [
        "path/to/fly/trial1/processed/dff.tif",
        "path/to/fly/trial2/processed/dff.tif"
    ]

    out_dir = "path/to/fly/processed"

    videos.make_multiple_video_raw_dff_beh(
        dffs,
        trial_dirs,
        out_dir,
        video_name="dff_beh_multiple.mp4",
        beh_dirs=None,  # define in case behavioural data is not located inside trial_dir
        sync_dirs=None,  # define in case sync data is not located inside trial_dir
        camera=5,
        stack_axes=[0, 1],
        greens=greens,
        reds=greens,
        share_lim=True,
        mask=None,
        share_mask=False,
        text=None,
        text_loc="dff",
        downsample=None,
        select_frames=None,
        time=True)
