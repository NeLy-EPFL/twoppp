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
    out_dir = "path/to/fly/processed"

    # make a video of just one trial
    videos.make_video_2p(
        green=greens[0],
        red=reds[0],
        video_name="raw_trial_1.mp4",
        out_dir=out_dir,
        trial_dir=trial_dirs[0])

    # make a video of multiple trials side by side
    videos.make_multiple_video_2p(
        greens=greens,
        reds=reds,
        out_dir=out_dir,
        video_name="raw_all_trials.mp4",
        trial_dir=trial_dirs[0])  # supply one trial dir to find frame rate
