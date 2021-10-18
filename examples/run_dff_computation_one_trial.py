from twoppp import dff
from twoppp.plot import videos

if __name__ == "__main__":
    trial_dir = "path/to/trial"
    stack = "path/to/trial/processed/green_denoised.tif"  # or green_com_warped.tif
    dff_dir = "path/to/trial/processed/dff.tif"
    video_dir = "path/to/trial/processed"
    dff_stack = dff.compute_dff_from_stack(
        stack,
        baseline_blur=10,
        baseline_med_filt=1,
        blur_pre=True,
        baseline_mode="convolve", # slow alternative: "quantile"
        baseline_length=10,
        baseline_quantile=0.05, # not used if 'convolve' is selected
        baseline_dir=None,
        min_baseline=None,
        dff_blur=0,
        dff_out_dir=dff_dir,
        return_stack=True)

    videos.make_video_dff(
        dff=dff_dir,
        out_dir=video_dir,
        video_name="dff.mp4",
        trial_dir=trial_dir,
        mask=None,
        text=None)
