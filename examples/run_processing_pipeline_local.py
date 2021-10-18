import os.path

from longterm import load
from longterm.pipeline import PreProcessFly, PreProcessParams

if __name__ == "__main__":
    fly_dir = os.path.join(load.NAS2_DIR_JB, "211005_J1M5", "Fly1")

    params = PreProcessParams()
    # supply some info about what kind of data has been recorded
    params.genotype = "J1M5"
    params.ball_tracking = "fictrac"  # "opflow", "fictrac", or None
    params.behaviour_as_videos = True  # whether 7 cam data is alread as video
    params.twop_scope = 2  # which of the two set-ups you used (1=LH&CLC, 2=FA+JB)

    # select which steps of the pipeline to perform
    params.use_com = True  # center of mass registration (COM)
    params.post_com_crop = True  # cropping after COM
    params.use_warp = True  # optic flow motion correction
    params.use_denoise = True  # denoising using deep interpolation
    params.use_dff = True  # computing dff
    params.use_df3d = False  # pose estimation
    params.make_dfs = True  # whether to generate synchronised dataframes
    params.use_df3dPostProcess = False  # pose post processing, e.g. computing angles
    params.make_dff_videos = False  # whether to make dff videos for each trial
    params.make_summary_stats = False  # whether to save mean/std/... for each trial

    # Define some paramters. There are much more that could be changed.
    # These are just the essentials
    params.post_com_crop_values = None  # cropping after com: (Y_SIZE, X_SIZE)
    params.denoise_crop_size = None  # cropping before denoising: (Y_SIZE, X_SIZE)
    params.overwrite = False  # whether to overwrite existing files with same name

    # make sure that the trial directories are selected correctly.
    # By default, every folder that does not contain "processed" is selected.
    # If this is different, then consider supplying a list of paths with the
    # 'trial_dirs' argument:
    #  e.g. ["path/to/fly_dir/trial_1", "path/to/fly_dir/trial_2"])
    # You could, for example, call the load.get_trials_from_fly() function
    # and use the 'startswith' and 'endswith' arguments.
    # Alternatively, you can save them in a file in the fly_dir called:
    # trial_dirs.txt
    # Also, make sure that the trial_dirs each contain the synchronisation data
    # and the behaviour data. If this is not the case, supply them separately
    # using the beh_trial_dirs and sync_trial_dirs arguments. Agains, these
    # can also be files called beh_trial_dirs.txt or sync_trial_dirs.txt
    print("Starting preprocessing of fly \n" + fly_dir)
    preprocess = PreProcessFly(fly_dir=fly_dir, params=params)
    preprocess.run_all_trials()
