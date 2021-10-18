from twoppp import dff
from twoppp.plot import videos

if __name__ == "__main__":

    trial_dirs = [
        "path/to/fly/trial1",
        "path/to/fly/trial2"
    ]

    stacks = [
        "path/to/fly/trial1/processed/green_denoised.tif",
        "path/to/fly/trial2/processed/green_denoised.tif"
    ]
    trial_baselines = [
        "path/to/fly/trial1/processed/dff_baseline.tif",
        "path/to/fly/trial2/processed/dff_baseline.tif"
    ]
    fly_baseline = [
        "path/to/fly/processed/dff_baseline.tif"
    ]

    dff_dirs = [
        "path/to/fly/trial1/processed/dff.tif",
        "path/to/fly/trial2/processed/dff.tif"
    ]
    dff_video_dir = "path/to/fly/processed"

    # find one common baseline across all stacks and save it
    dff_baseline = dff.find_dff_baseline_multi_stack_load_single(
        stacks,
        trial_baselines,
        baseline_blur=10,
        baseline_med_filt=1,
        blur_pre=True,
        baseline_length=10,
        baseline_quantile=0.05,
        baseline_dir=fly_baseline,
        min_baseline=None)

    for i_trial, (stack, dff_dir) in enumerate(stacks, dff_dirs):
        dff_stack = dff.compute_dff_from_stack(
            stack,
            baseline_blur=0,
            baseline_med_filt=1,
            blur_pre=False,
            baseline_mode="fromfile",
            baseline_length=10,
            baseline_quantile=0.05,
            baseline_dir=fly_baseline,
            min_baseline=None,
            dff_out_dir=dff_dir,
            return_stack=True)

    videos.make_multiple_video_dff(dffs=dff_dirs,
        out_dir=dff_video_dir,
        video_name="dff_multiple.tif",
        trial_dir=trial_dirs[0],  # supply example trial dir to find frame rate
        share_lim=True,
        mask=None,
        share_mask=False,
        text=None)
