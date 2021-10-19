# Jonas Braun
# jonas.braun@epfl.ch
# 17.03.2021

import os, sys
from os.path import join
import gc
import numpy as np
from copy import deepcopy
import time
import pickle
from shutil import copy2
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt

from ofco.utils import default_parameters
from deepinterpolation import interface as denoise

from twoppp import load, dff, OUTPUT_PATH
from twoppp.utils import makedirs_safe, get_stack, save_stack, readlines_tolist
from twoppp.register import warping
from twoppp.denoise import prepare_corrected_data
from twoppp.behaviour import df3d
from twoppp.plot.videos import make_video_dff, make_multiple_video_dff, make_video_raw_dff_beh, make_multiple_video_raw_dff_beh
from twoppp.rois import local_correlations, get_roi_signals_df
from twoppp.behaviour.synchronisation import get_synchronised_trial_dataframes
from twoppp.behaviour.optic_flow import get_opflow_df, get_opflow_in_twop_df
from twoppp.behaviour.fictrac import get_fictrac_df

class PreProcessParams:
    """Class containing all default parameters for the PreProcessFly class.
    Usage:
    param = PreProcessParams()
    param.value1 = ABC
    param.value2 = DEF
    """

    def __init__(self):
        """Class containing all default parameters for the PreProcessFly class."""
        self.genotype = " "
        # names of files generated throughout processing
        self.red_raw = "red.tif"
        self.green_raw = "green.tif"
        self.green_com_crop = "green_com_crop.tif"
        self.red_com_crop = "red_com_crop.tif"
        self.red_warped = "red_warped.tif"
        self.green_warped = "green_warped.tif"
        self.red_com_warped = "red_com_warped.tif"
        self.green_com_warped = "green_com_warped.tif"
        self.green_denoised = "green_denoised.tif"
        self.red_denoised = "red_denoised.tif"
        self.dff = "dff.tif"
        self.dff_baseline = "dff_baseline.tif"
        self.drr = "drr.tif"
        self.drr_baseline = "drr_baseline.tif"
        self.dff_mask = "dff_mask.tif"

        self.ref_frame = "ref_frame_com.tif"
        self.com_offset = "com_offset.npy"
        self.motion_field = "w.npy"

        self.trial_dirs = "trial_dirs.txt"
        self.beh_trial_dirs = "beh_trial_dirs.txt"
        self.sync_trial_dirs = "sync_trial_dirs.txt"

        self.summary_stats = "compare_trials.pkl"

        self.opflow_df_out_dir = "opflow_df.pkl"
        self.df3d_df_out_dir = "beh_df.pkl"
        self.twop_df_out_dir = "twop_df.pkl"

        self.roi_centers = "ROI_centers.txt"
        self.roi_mask = "ROI_mask.tif"

        # mode of pre-processing: if True, perfom one processing step
        # on each trial before moving to next processing step
        self.breadth_first = True
        self.overwrite = False  # whether to overwrite existing files with same name
        self.use_warp = False  # optic flow motion correction
        self.use_denoise = False  # denoising using deep interpolation
        self.use_dff = False  # computing delta F / F
        self.use_df3d = False  # pose estimation
        self.use_df3dPostProcess = False  # pose post processing, e.g. computing angles
        self.use_behaviour_classifier = False  # use behavioural classifier
        self.make_dfs = False  # whether to generate synchronised dataframes
        self.cleanup_files = False
        self.make_dff_videos = False  # whether to make dff videos for each trial
        self.make_summary_stats = True  # whether to save mean/std/... for each trial
        self.ball_tracking = "opflow"  # "opflow", "fictrac", or None
        self.add_df3d_to_df = False  # whether to add pose estimation results to data frame

        # ofco (optical flow motion correction) params
        self.i_ref_trial = 0  # in which trial to search for reference frame
        self.i_ref_frame = 0  # which frame to pick in that trial
        self.use_com = True  # whethter to use center of mass registration
        self.post_com_crop = True  # whether to crop the images after COM registration
        self.post_com_crop_values = None  # cropping after com: (Y_SIZE, X_SIZE)
        self.save_motion_field = False  # whether to save motion field. They are very large
        self.ofco_param = default_parameters()  # parameters from ofco including regularisation
        self.ofco_parallel = True  # whether to use parallel processing with multiprocessing.Pool
        self.ofco_verbose = True  # whether to inform about intermediate processing steps
        # self.ofco_out_dtype = np.float32  # TODO: verify whether this is the best choice

        # denoising params for DeepInterpolation
        self.denoise_crop_size = None  # cropping before denoising: (Y_SIZE, X_SIZE)
        self.denoise_crop_offset = (None, None)  # offset of the crop: (Y_OFFSET, X_OFFSET)
        # the following three serve to make local copies of the data to allow for faster processing
        self.denoise_tmp_data_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation","data")
        self.denoise_tmp_data_name = self._make_denoise_data_name
        self.denoise_tmp_run_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation", "runs")
        self.denoise_runid = self._make_runid
        self.denoise_final_dir = "denoising_run"  # name of folder to save model in
        self.denoise_delete_tmp_run_dir = True  # whether to delete the temporary run directory
        self.denoise_params = denoise.DefaultInterpolationParams()
        self.denoise_train_each_trial = False  # if False, only train on one trial.
        self.denoise_train_trial = 0  # which trial to train on
        # whether to use correction of illumination along medial-lateral axis before denoising
        self.denoise_correct_illumination_leftright = False

        # dff params
        self.dff_common_baseline = True  # whether to use the same baseline across all trials
        self.dff_baseline_blur = 10  # sigma of spatial Gaussian filter applied to compute baseline
        self.dff_baseline_med_filt = 1  # median filter applied to compute basline
        # whether to apply filters on stack before baseline computation or on baseline
        self.dff_baseline_blur_pre = True  
        self.dff_baseline_mode = "convolve"  # slower version: "quantile"
        self.dff_baseline_length = 10  # length in samples of moving average filter for baseline
        self.dff_baseline_quantile = 0.05  # only used is baseline mode is "quantile"
        self.dff_use_crop=False  # whether to crop dff
        self.dff_manual_add_to_crop = 20  # only used if automatic crop is selected
        self.dff_blur = 0  # whether to blur the dff after computing it. supply width of Gaussian
        self.dff_min_baseline = None  # value of minimum baseline. e.g., 0 to prevent neg. baseline
        self.dff_baseline_exclude_trials = None  # list with faulty trials to be excluded

        self.dff_video_name = "dff"
        self.dff_beh_video_name = "dff_beh"
        self.dff_video_pmin = None  # percentile minimum for dff videos
        self.dff_video_vmin = 0  # absolute minimum for dff videos
        self.dff_video_pmax = 99  # percentile maximum for dff videos
        self.dff_video_vmax = None  # absolute maximum for dff videos
        self.dff_video_share_lim = True  # whether to share dff limit across trials in one video
        self.dff_video_log_lim = False  # whether to use a logarithmic scale/limit in video
        self.dff_video_downsample = 2  # temporal downsampling of dff videos to reduce size
        self.default_video_camera = 6  # which video camera to use. 5 for set-up 2 and 6 for setup 1

        # deepfly3d params
        self.behaviour_as_videos = True  # whether 7 cam data is available as .mp4 videos
        self.twop_scope = 2  # which of the two set-ups you used (1=LH&CLC, 2=FA+JB)

        # optic flow params
        self.opflow_win_size = 80  # moving average window used to smooth optical flow
        self.thres_walk = 0.03  # absolute threshold on optical flow data for walking. unit rot/s
        self.thres_rest = 0.01  # absolute threshold on optical flow data for resting. unit rot/s

        # ROI extraction params
        self.roi_size = (7,11)  # size of pattern for ROI extraction
        self.roi_pattern = "default"  # shape of pattern. "default" gives a rhomboid

    @staticmethod
    def _make_runid(processed_dir, appendix=None):
        trial_dir, _ = os.path.split(processed_dir)
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = date + "_" + fly + "_" + trial
        return (out + "_" + appendix) if appendix is not None else out

    def _make_denoise_data_name(self, processed_dir):
        return self._make_runid(processed_dir, appendix=self.green_com_warped)

class PreProcessFly:
    """Class to automatise preprocessing of 2p and behavioural data."""

    def __init__(self, fly_dir, params=PreProcessParams(), trial_dirs=None, selected_trials=None,
                 beh_trial_dirs=None, sync_trial_dirs=None):
        """Class to automatise pre-processing of two-photon and behavioural data.
        All parameters and which pre-processing steps to perform can be selected by supplying params

        Parameters
        ----------
        fly_dir : str
            base directory that contains trials

        params : PreProcessParams, optional
            parameter class including processing parameters and which processing steps to perform,
            by default PreProcessParams()

        trial_dirs : list of str or str, optional
            directories containing the two photon data
            if None: automatic. find all dirs in fly_dir: load.get_trials_from_fly(self.fly_dir)[0],
            if "fromfile": read lines of fly_dir/"params.trial_dirs" file (default: trial_dirs.txt)
            if str: supply the path to a text file containint a trial directory on each list
            otherwise: supply a list of trials
            by default None

        selected_trials : list of int, optional
            list of indices of trials to keep. all others will be excluded from analysis,
            by default None

        beh_trial_dirs : list of str or str, optional
            directories containing behavioural data, i.e., 7 cam, optic flow,
            same system as for trial_dirs, but if None, beh_trial_dirs=trial_dirs will be selected,
            by default None

        sync_trial_dirs : ist of str or str, optional
            directories containing thorsync data
            same system as for trial_dirs, but if None, sync_trial_dirs=trial_dirs will be selected,
            by default None

        Raises
        ------
        FileNotFoundError
            if trial_dirs/beh_trial_dirs/sync_trial_dirs file is not found

        NotImplementedError
            if selected trials is not None and not a list of integers
        """
        super().__init__()
        self.params = params
        self.fly_dir = fly_dir
        self.selected_trials = selected_trials

        # get the trial directories containing two-photon data
        if trial_dirs == "fromfile":
            self.trial_dirs = readlines_tolist(join(self.fly_dir, self.params.trial_dirs))
        elif isinstance(trial_dirs, str) and os.path.isfile(trial_dirs):
            self.trial_dirs = readlines_tolist(trial_dirs)
        elif isinstance(trial_dirs, str):
            raise FileNotFoundError("could not find: " + trial_dirs)
        elif isinstance(trial_dirs, list):
            self.trial_dirs = trial_dirs
        else:
            print("Automatically detecting trial dirs.")
            self.trial_dirs = load.get_trials_from_fly(self.fly_dir)[0]

        # get the behavioural data directories in case they are not the same as the two photon ones
        if beh_trial_dirs == "fromfile":
            self.beh_trial_dirs = readlines_tolist(join(self.fly_dir, self.params.beh_trial_dirs))
        elif isinstance(beh_trial_dirs, str) and os.path.isfile(beh_trial_dirs):
            self.beh_trial_dirs = readlines_tolist(beh_trial_dirs)
        elif isinstance(beh_trial_dirs, str):
            raise FileNotFoundError("could not find: " + beh_trial_dirs)
        elif isinstance(beh_trial_dirs, list):
            self.beh_trial_dirs = beh_trial_dirs
        else:
            self.beh_trial_dirs = self.trial_dirs

        # get the thorsync data directories in case they are not the same as the two photon ones
        if sync_trial_dirs == "fromfile":
            self.sync_trial_dirs = readlines_tolist(join(self.fly_dir, self.params.sync_trial_dirs))
        elif isinstance(sync_trial_dirs, str) and os.path.isfile(sync_trial_dirs):
            self.sync_trial_dirs = readlines_tolist(sync_trial_dirs)
        elif isinstance(sync_trial_dirs, str):
            raise FileNotFoundError("could not find: " + sync_trial_dirs)
        elif isinstance(sync_trial_dirs, list):
            self.sync_trial_dirs = sync_trial_dirs
        else:
            self.sync_trial_dirs = self.trial_dirs

        # select the trials according to selected_trials
        self._match_trials_and_beh_trials()
        if self.selected_trials is not None and isinstance(self.selected_trials, list):
            self.trial_dirs = [self.trial_dirs[i] for i in self.selected_trials]
            self.beh_trial_dirs = [self.beh_trial_dirs[i] for i in self.selected_trials]
            self.sync_trial_dirs = [self.sync_trial_dirs[i] for i in self.selected_trials]
            self._match_trials_and_beh_trials()
        elif self.selected_trials is not None:
            raise NotImplementedError("selected_trials should be None or a list of ints.")
        else:
            self.selected_trials = np.arange(len(self.trial_dirs))
        
        self._get_trial_names()
        self._get_experiment_info()
        self._create_processed_structure()
        # self._save_ref_frame() --> moved to run_all_trials

    def _get_trial_names(self):
        """get the trial name of each trial by splitting the direcory
        and taking the last folder name as trial name
        """
        trial_names = []
        for trial in self.trial_dirs:
            _, name = os.path.split(trial)
            trial_names.append(name)
        self.trial_names = trial_names

    def _get_experiment_info(self):
        """extract information about the expteriment from the fly_dir
        this includes:
            fly: the number of the fly on that day
            date: the date of the recording
            genotype: the genotype of the fly recorded
        """
        try:
            tmp, fly = os.path.split(self.fly_dir)
            self.fly = int(fly[-1:])
        except:
            self.fly = 0
        try:
            _, date = os.path.split(tmp)
            self.date = int(date[:6])
        except:
            self.date = 123456
        try:
            if len(date) > 7:
                self.genotpye = date[7:]
            else:
                self.genotpye = self.params.genotype
        except:
            self.genotpye = self.params.genotype

    def _match_trials_and_beh_trials(self):
        """check that the same number of two-photon, behavioural and sync trials exist
        """
        assert len(self.beh_trial_dirs) == len(self.trial_dirs)
        assert len(self.sync_trial_dirs) == len(self.trial_dirs)
        return

    def _create_processed_structure(self):
        """create a "processed" sub-folder in each trial_dir and one inside the fly_dir.
        the first one will contain trial-specific data, the second one fly_specific data
        """
        self.fly_processed_dir = join(self.fly_dir, load.PROCESSED_FOLDER)
        makedirs_safe(self.fly_processed_dir)
        self.trial_processed_dirs = [join(trial_dir, load.PROCESSED_FOLDER)
                                     if trial_dir != "" else ""
                                     for trial_dir in self.trial_dirs]
        _ = [makedirs_safe(processed_dir)
             for processed_dir in self.trial_processed_dirs if processed_dir != ""]

    def _save_ref_frame(self):
        """save the reference frame for optical flow motion registration
        uses the following params:
            ref_frame: if "", don't compute reference frame, else use as file name
            i_ref_trial: which trial to use
            i_ref_frame: which frame to use
            raw_red: name of raw data file
            use_com: whether to use center of mass registration
            post_com_crop: whther to crop after com registration
            post_com_crop_values: how to crop
            overwrite: whether to overwrite existing files
        """
        if self.params.ref_frame == "":
            return
        ref_trial_dir = self.trial_dirs[self.params.i_ref_trial]
        ref_processed_dir = self.trial_processed_dirs[self.params.i_ref_trial]
        _ = load.convert_raw_to_tiff(ref_trial_dir,
                                     overwrite=self.params.overwrite,
                                     return_stacks=False)
        ref_stack = join(ref_processed_dir, self.params.red_raw)
        self.ref_frame = join(self.fly_processed_dir, self.params.ref_frame)
        #TODO: leave a note which ref frame is saved
        warping.save_ref_frame(stack=ref_stack,
                               ref_frame_dir=self.ref_frame,
                               i_frame=self.params.i_ref_frame,
                               com_pre_reg=self.params.use_com,
                               overwrite=self.params.overwrite,
                               crop=self.params.post_com_crop_values 
                               if self.params.post_com_crop else None)

    def run_all_trials(self):
        """run pre-processing for all trials
        according to the steps and parameters defined in self.params.
        Either runs breadth first (finish one step for all trials)
        or depth dirst (go as far as possible with one trial before touching the others).
        uses the following params:
            breadth_first: to decide whether to run breath first or depth first
        """
        self._save_ref_frame()
        if self.params.breadth_first:
            self._run_breadth_first()
        else:
            self._run_depth_first()

    def run_single_trial(self, i_trial):
        """run all processing steps defined in self.params for one trial
        uses the following params to decide which steps to perform:
            use_warp
            use_denoise
            use_dff
            use_df3d
            use_df3dPostProcess
            make_dff_videos
            make_summary_stats

        Parameters
        ----------
        i_trial : int
            which trial to run the processing for
        """
        self._save_ref_frame()
        trial_dir = self.trial_dirs[i_trial]
        beh_trial_dir = self.beh_trial_dirs[i_trial]
        processed_dir = self.trial_processed_dirs[i_trial]
        self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
        if self.params.use_warp:
            self._warp_trial(processed_dir)
        if self.params.use_denoise:
            self._denoise_trial_trainNinfer(processed_dir)
        if self.params.use_dff:
            self._compute_dff_trial(processed_dir, force_single_baseline=True)
        if self.params.use_df3d:
            self._pose_estimate(beh_trial_dir)
        if self.params.use_df3dPostProcess:
            self._post_process_pose(beh_trial_dir)
        if self.params.make_dff_videos:
            self._make_dff_video_trial(i_trial)
        if self.params.make_summary_stats:
            self._compute_summary_stats(i_trials=[i_trial])

    def _run_breadth_first(self):
        """run all trials and finish one stage for all trials before going to next stage
        uses the following params to decide which steps to perform:
            use_warp
            use_com
            use_denoise
            use_dff
            use_df3d
            use_df3dPostProcess
            make_dff_videos
            make_summary_stats
            make_dfs
        """
        for trial_dir, processed_dir in \
            zip(self.trial_dirs, self.trial_processed_dirs):
            print(time.ctime(time.time()), " converting trial to tif: " + trial_dir)
            self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
            gc.collect()
        if self.params.use_warp:
            _ = [self._warp_trial(processed_dir)
                 for processed_dir in self.trial_processed_dirs]
            gc.collect()
        elif self.params.use_com:
            _ = [self._com_correct_trial(processed_dir)
                 for processed_dir in self.trial_processed_dirs]
        if self.params.use_denoise:
            self._denoise_all_trials()
        if self.params.use_dff:
            print(time.ctime(time.time()), " computing dff")
            self._compute_dff_alltrials()
        if self.params.use_df3d:
            print(time.ctime(time.time()), " performing pose estimation with deepfly3d")
            self._pose_estimate()
        if self.params.use_df3dPostProcess:
            print(time.ctime(time.time()), " post processing df3d pose")
            self._post_process_pose()
        if self.params.make_dff_videos:
            self._make_dff_videos()
        if self.params.make_summary_stats:
            if self.params.denoise_correct_illumination_leftright:
                self._compute_summary_stats(force_overwrite=True)
            else:
                self._compute_summary_stats()
        if self.params.make_dfs:
            self.get_dfs()

    def _run_depth_first(self, force_single_trial_dff=False):
        """run processing for all trials and going as far as possible with one trial
        before considering other trials
        uses the following params to decide which steps to perform:
            use_warp
            use_denoise
            use_dff
            use_df3d
            use_df3dPostProcess
            make_dff_videos
            make_summary_stats
            make_dfs

        Parameters
        ----------
        force_single_trial_dff : bool, optional
            compute dff for single trials and don't use common baseline,
            allows to compute it before other trials are finished. by default False
        """
        for i_trial, (trial_dir, beh_trial_dir, processed_dir) in \
            enumerate(zip(self.trial_dirs, self.beh_trial_dirs, self.trial_processed_dirs)):
            self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
            if self.params.use_warp:
                self._warp_trial(processed_dir)
            if self.params.use_denoise:
                self._denoise_trial_trainNinfer(processed_dir)
            if self.params.use_df3d:
                self._pose_estimate(trial_dir)
            if self.params.use_dff and force_single_trial_dff:
                self._compute_dff_trial(processed_dir, force_single_baseline=True)
            if self.params.use_df3d:
                self._pose_estimate(beh_trial_dir)
            if self.params.use_df3dPostProcess:
                self._post_process_pose(beh_trial_dir)
        if self.params.use_dff:
            self._compute_dff_alltrials()
        if self.params.make_dff_videos:
            self._make_dff_videos()
        if self.params.make_summary_stats:
            if self.params.denoise_correct_illumination_leftright:
                self._compute_summary_stats(force_overwrite=True)
            else:
                self._compute_summary_stats()
        if self.params.make_dfs:
            self.get_dfs()

    def _convert_raw_to_tiff_trial(self, trial_dir, processed_dir):
        """first step of processing: convert .raw files into .tif
        uses the following params:
            overwrite
            green_raw
            red_raw
        Parameters
        ----------
        trial_dir : str
            directory where to find the .raw data
        processed_dir : str
            directory where to store the .tif data
        """
        if trial_dir != "" and processed_dir != "":
            _ = load.convert_raw_to_tiff(trial_dir,
                                        overwrite=self.params.overwrite,
                                        return_stacks=False,
                                        green_dir=join(processed_dir, self.params.green_raw),
                                        red_dir=join(processed_dir, self.params.red_raw)
                                        )

    def _com_correct_trial(self, processed_dir):
        """apply center of mass correction to a trial
        uses the following params:
            red_raw
            green_raw
            red_com_crop
            green_com_crop
            post_com_crop_values
            com_offset
            overwrite

        Parameters
        ----------
        processed_dir : str
            processed directory of that trial containing raw tif and where com corrected
            tif will be saved to
        """
        print(time.ctime(time.time()), " com correcting and cropping trial: " + processed_dir)
        _ = warping.center_and_crop(stack1=join(processed_dir, self.params.red_raw),
                                    stack2=join(processed_dir, self.params.green_raw),
                                    crop=self.params.post_com_crop_values,
                                    stack1_out_dir=join(processed_dir, self.params.red_com_crop),
                                    stack2_out_dir=join(processed_dir, self.params.green_com_crop),
                                    offset_dir=join(processed_dir, self.params.com_offset),
                                    return_stacks=False,
                                    overwrite=self.params.overwrite)

    def _warp_trial(self, processed_dir):
        """apply optic flow motion registration
        (and potentialy center of mass registration before)
        uses the following params:
            use_com
            post_com_crop
            red_raw
            green_raw
            red_com_crop
            green_com_crop
            red_com_warped
            green_com_warped
            com_offset
            overwrite
            ofco_parallel
            ofco_verbose
            motion_field
            save_motion_field
            ofco_param
        Parameters
        ----------
        processed_dir : str
            processed directory of that trial containing raw tif and where com corrected
            tif will be saved to
        """
        if processed_dir != "":
            if self.params.use_com and self.params.post_com_crop:
                # if to be cropped, perform COM outside of the warping.warp() call
                # otherwise, perform COM inside warping.warp
                self._com_correct_trial(processed_dir)
                if self.params.cleanup_files:
                    pass
                    # TODO: os.remove(join(processed_dir, self.params.red_raw))
                    # TODO: os.remove(join(processed_dir, self.params.green_raw))
                stack1 = join(processed_dir, self.params.red_com_crop)
                stack2 = join(processed_dir, self.params.green_com_crop)
                use_com = False  # because it was already done
            else:
                stack1 = join(processed_dir, self.params.red_raw)
                stack2 = join(processed_dir, self.params.green_raw)
                use_com = self.params.use_com

            print(time.ctime(time.time()), "warping trial: " + processed_dir)

            _ = warping.warp(stack1=stack1,
                            stack2=stack2,
                            ref_frame=self.ref_frame,
                            stack1_out_dir=join(processed_dir, self.params.red_com_warped),
                            stack2_out_dir=join(processed_dir, self.params.green_com_warped),
                            com_pre_reg=use_com,
                            offset_dir=join(processed_dir, self.params.com_offset),
                            return_stacks=False,
                            overwrite=self.params.overwrite,
                            select_frames=None,
                            parallel=self.params.ofco_parallel,
                            verbose=self.params.ofco_verbose,
                            w_output=join(processed_dir, self.params.motion_field),
                            initial_w=None,
                            save_motion_field=self.params.save_motion_field,
                            param=self.params.ofco_param
                            )

    def _denoise_trial_trainNinfer(self, processed_dir):
        """train a DeepInterpolation model on one trial and infer on the same trial
        uses the following params:
            green_com_warped
            green_denoised
            overwrite
            denoise_tmp_data_dir
            denoise_tmp_data_name
            denoise_crop_offset
            denoise_crop_size
            denoise_tmp_run_dir
            denoise_runid
            denoise_params
            denoise_final_dir
            denoise_delete_tmp_run_dir
        Parameters
        ----------
        processed_dir : str
            processed directory of that trial containing motion corrected tif and where denoised
            tif will be saved to
        """
        if processed_dir != "":
            print(time.ctime(time.time()), "denoising trial: " + processed_dir)
            if os.path.isfile(join(processed_dir, self.params.green_denoised)) \
                and not self.params.overwrite:
                return
            input_data = join(processed_dir, self.params.green_com_warped)
            tmp_data_dir = join(self.params.denoise_tmp_data_dir,
                                self.params.denoise_tmp_data_name(processed_dir))
            denoise.prepare_data(train_data_tifs=input_data,
                                out_data_tifs=tmp_data_dir,
                                offset=self.params.denoise_crop_offset,
                                size=self.params.denoise_crop_size)
            tmp_run_dir = denoise.train(train_data_tifs=tmp_data_dir,
                                        run_base_dir=self.params.denoise_tmp_run_dir,
                                        run_identifier=self.params.denoise_runid(processed_dir),
                                        params=self.params.denoise_params)
            denoise.inference(data_tifs=tmp_data_dir,
                            run_dir=tmp_run_dir,
                            tif_out_dirs=join(processed_dir, self.params.green_denoised),
                            params=self.params.denoise_params)
            denoise.clean_up(tmp_run_dir, tmp_data_dir)
            denoise.copy_run_dir(tmp_run_dir,
                                 join(processed_dir, self.params.denoise_final_dir),
                                 delete_tmp=self.params.denoise_delete_tmp_run_dir)
            gc.collect()

    def _denoise_all_trials(self):
        """denoise all trials using DeepInterpolation either by training on one trial or on all.
        Using the following params:
            green_com_warped
            green_denoised
            overwrite
            denoise_correct_illumination_leftright
            summary_stats
            denoise_train_trial
            denoise_crop_offset
            denoise_crop_size
            denoise_params
            denoise_tmp_data_dir
            denoise_tmp_data_name
            denoise_final_dir
            denoise_delete_tmp_run_dir
        """
        if self.params.denoise_train_each_trial:
            print(time.ctime(time.time()), "Start denoising by training on all trials.")
            _ = [self._denoise_trial_trainNinfer(processed_dir)
                 for processed_dir in self.trial_processed_dirs]
        else:
            print(time.ctime(time.time()),
                  f"Start denoising by training on one trial: {self.params.denoise_train_trial}")
            input_datas = []
            tmp_data_dirs = []
            if os.path.isdir(join(self.fly_processed_dir, self.params.denoise_final_dir)) \
                and not self.params.overwrite:
                tmp_run_dir = join(self.fly_processed_dir, self.params.denoise_final_dir)
                already_trained = True
                print("Using already trained model.")
            else:
                already_trained = False

            already_denoised_trials = [
                os.path.isfile(join(processed_dir, self.params.green_denoised))
                for processed_dir in self.trial_processed_dirs]
            if all(already_denoised_trials) and not self.params.overwrite:
                return
            elif any(already_denoised_trials) and already_trained and not self.params.overwrite:
                todo_trial_processed_dirs = [
                    processed_dir for processed_dir, already_denoised
                    in zip(self.trial_processed_dirs, already_denoised_trials)
                    if not already_denoised]
            else:
                todo_trial_processed_dirs = self.trial_processed_dirs

            input_datas = [join(processed_dir, self.params.green_com_warped)
                            for processed_dir in todo_trial_processed_dirs]
            tmp_data_dirs = [join(self.params.denoise_tmp_data_dir,
                                self.params.denoise_tmp_data_name(processed_dir))
                            for processed_dir in todo_trial_processed_dirs]

            if self.params.denoise_correct_illumination_leftright:
                self._compute_summary_stats(raw_only=True, force_overwrite=True)

                prepare_corrected_data(train_data_tifs=input_datas,
                                       out_data_tifs=tmp_data_dirs,
                                       fly_dir=self.fly_dir,
                                       summary_dict_pickle=join(self.fly_processed_dir,
                                                                self.params.summary_stats))
            else:
                denoise.prepare_data(train_data_tifs=input_datas,
                                    out_data_tifs=tmp_data_dirs,
                                    offset=self.params.denoise_crop_offset,
                                    size=self.params.denoise_crop_size)

            if not already_trained:
                training_processed_dir = todo_trial_processed_dirs[self.params.denoise_train_trial]
                training_tmp_data_dir = tmp_data_dirs[self.params.denoise_train_trial]

                with mp.Manager() as manager:
                    print(time.ctime(time.time()),
                          "Starting separate process to train denoising model.")
                    share_dict = manager.dict()
                    kwargs = {
                        "train_data_tifs": training_tmp_data_dir,
                        "run_base_dir": self.params.denoise_tmp_run_dir,
                        "run_identifier": self.params.denoise_runid(training_processed_dir),
                        "params": self.params.denoise_params,
                        "return_dict_run_dir": share_dict
                    }
                    p = mp.Process(target=denoise.train, kwargs=kwargs)
                    p.start()
                    p.join()
                    tmp_run_dir = share_dict[0]

            tif_out_dirs = [join(processed_dir, self.params.green_denoised)
                            for processed_dir in todo_trial_processed_dirs]
            kwargs = {
                "data_tifs": tmp_data_dirs,
                "run_dir": tmp_run_dir,
                "tif_out_dirs": tif_out_dirs,
                "params": self.params.denoise_params
            }
            print(time.ctime(time.time()), "Starting separate process to perform inference.")
            p = mp.Process(target=denoise.inference, kwargs=kwargs)
            p.start()
            p.join()
            denoise.clean_up(tmp_run_dir, tmp_data_dirs)
            if not already_trained:
                denoise.copy_run_dir(tmp_run_dir,
                                    join(self.fly_processed_dir, self.params.denoise_final_dir),
                                    delete_tmp=self.params.denoise_delete_tmp_run_dir)
            gc.collect()

    def _compute_dff_trial(self, processed_dir, force_single_baseline=False, force_overwrite=False):
        """Compute Delta F/F for one trial
        If you want to compute dff for not-denoised data, rename params.green_denoised to however
        your not-denoised data is called (and ideally change the name of params.dff as well).
        Uses the following params:
            green_denoised
            dff_baseline
            dff
            overwrite
            dff_common_baseline
            dff_baseline
            dff_baseline_mode
            dff_baseline_blur_pre
            dff_baseline_blur
            dff_baseline_med_filt
            dff_baseline_length
            dff_baseline_quantile
            dff_use_crop
            dff_manual_add_to_crop
            dff_blur
            dff_min_baseline

        Parameters
        ----------
        processed_dir : str
            processed directory of that trial containing motion corrected (and denoised) tif
            and where dff tif will be saved to

        force_single_baseline : bool, optional
            whether to use a baseline that was only calculated on this trial.
            If False load the common baseline from file, by default False

        force_overwrite : bool, optional
            possibility to force overwriting of files if params.overwrite == False, by default False
        """
        if processed_dir != "":
            stack = join(processed_dir, self.params.green_denoised)
            if self.params.dff_common_baseline and not force_single_baseline:
                baseline_mode = "fromfile"
                blur_pre = False
                baseline_blur = 0
                baseline_med_filt = 0
                baseline_dir = join(self.fly_processed_dir, self.params.dff_baseline)
            else:
                baseline_mode = self.params.dff_baseline_mode
                blur_pre = self.params.dff_baseline_blur_pre
                baseline_blur = self.params.dff_baseline_blur
                baseline_med_filt = self.params.dff_baseline_med_filt
                baseline_dir = join(processed_dir, self.params.dff_baseline)

            if not os.path.isfile(join(processed_dir, self.params.dff)) \
                or self.params.overwrite or force_overwrite:
                _ = dff.compute_dff_from_stack(
                    stack,
                    baseline_blur=baseline_blur,
                    baseline_med_filt=baseline_med_filt,
                    blur_pre=blur_pre,
                    baseline_mode=baseline_mode,
                    baseline_length=self.params.dff_baseline_length,
                    baseline_quantile=self.params.dff_baseline_quantile,
                    baseline_dir=baseline_dir,
                    use_crop=self.params.dff_use_crop,
                    manual_add_to_crop=self.params.dff_manual_add_to_crop,
                    dff_blur=self.params.dff_blur,
                    min_baseline=self.params.dff_min_baseline,
                    dff_out_dir=join(processed_dir, self.params.dff),
                    return_stack=False)

    def _compute_dff_alltrials(self):
        """Compute delta F / F for all trials
        If you want to compute dff for not-denoised data, rename params.green_denoised to however
        your not-denoised data is called (and ideally change the name of params.dff as well).
        Uses the following params:
            green_denoised
            dff_baseline
            dff
            overwrite
            dff_baseline_exclude_trials
            dff_common_baseline
            dff_baseline
            dff_baseline_mode
            dff_baseline_blur_pre
            dff_baseline_blur
            dff_baseline_med_filt
            dff_baseline_length
            dff_baseline_quantile
            dff_min_baseline
        """
        if self.params.dff_baseline_exclude_trials is None:
            self.params.dff_baseline_exclude_trials = [False for _ in self.trial_dirs]
        stacks = [join(processed_dir, self.params.green_denoised)
                  for i_dir, processed_dir in enumerate(self.trial_processed_dirs)
                  if processed_dir != "" and not self.params.dff_baseline_exclude_trials[i_dir]]
        baseline_dirs = [
            join(processed_dir, self.params.dff_baseline)
            for i_dir, processed_dir in enumerate(self.trial_processed_dirs)
            if processed_dir != "" and not self.params.dff_baseline_exclude_trials[i_dir]]
        if self.params.dff_common_baseline:
            print(time.ctime(time.time()), "computing dff baseline")
            if not os.path.isfile(join(self.fly_processed_dir, self.params.dff_baseline)) \
                or self.params.overwrite:
                _ = dff.find_dff_baseline_multi_stack_load_single(stacks=stacks,
                        individual_baseline_dirs=baseline_dirs,
                        baseline_blur=self.params.dff_baseline_blur,
                        baseline_med_filt=self.params.dff_baseline_med_filt,
                        blur_pre=self.params.dff_baseline_blur_pre,
                        baseline_mode=self.params.dff_baseline_mode,
                        baseline_length=self.params.dff_baseline_length,
                        baseline_quantile=self.params.dff_baseline_quantile,
                        min_baseline=self.params.dff_min_baseline,
                        baseline_dir=join(self.fly_processed_dir, self.params.dff_baseline))

        for processed_dir in self.trial_processed_dirs:
            print(time.ctime(time.time()), "computing dff for trial: " + processed_dir)
            self._compute_dff_trial(processed_dir)

    def _pose_estimate(self, trial_dirs=None):
        """run pose estimation using DeepFly3D.
        Uses the following params:
            behaviour_as_videos
            twop_scope
            overwrite
        Parameters
        ----------
        trial_dirs : list of str, optional
            which trials to run deepfly3d for. if not supplied, use self.beh_trial_dirs,
            by default None
        """
        trial_dirs = deepcopy(self.beh_trial_dirs) if trial_dirs is None else trial_dirs
        if not isinstance(trial_dirs, list):
            trial_dirs = [trial_dirs]
        trial_dirs = [trial_dir for trial_dir in trial_dirs if trial_dir != ""]
        tmp_dir = df3d.prepare_for_df3d(trial_dirs=trial_dirs,
                                        videos=self.params.behaviour_as_videos,
                                        scope=self.params.twop_scope,
                                        overwrite=self.params.overwrite
                                        )
        df3d.run_df3d(tmp_dir)

    def _post_process_pose(self, trial_dirs=None):
        """run deepfly3d pose post processding.
        this aligns the positions to a reference fly and computes the joint angles.
        Uses the following params:
            overwrite

        Parameters
        ----------
        trial_dirs : [type], optional
            [description], by default None
        """
        trial_dirs = deepcopy(self.beh_trial_dirs) if trial_dirs is None else trial_dirs
        for trial_dir in trial_dirs:
            if trial_dir != "":
                print(time.ctime(time.time()), "postprocessing df3d for trial: " + trial_dir)
                df3d.postprocess_df3d_trial(trial_dir, overwrite=self.params.overwrite)

    def _make_dff_video_trial(self, i_trial, mask=None):
        """make a video of the DFF for one trial.
        Uses the following parameters:
            dff_video_name
            overwrite
            dff_mask
            overwrite
            dff
            dff_baseline
            dff_video_vmin
            dff_video_vmax
            dff_video_pmin
            dff_video_pmax

        Parameters
        ----------
        i_trial : int
            index of trial in self.trial_dirs

        mask : bool or np.array or str, optional
            if numpy array, use it directly as mask to make vide.
            if "fromfile", load the mask specified at fly_processed_dir/self.params.dff_mask
            if bool and True, automatically find a mask (NOT RECOMMENDED!!!),
            by default None
        """
        processed_dir = self.trial_processed_dirs[i_trial]
        trial_dir = self.trial_dirs[i_trial]
        trial_name = self.trial_names[i_trial]
        if not os.path.isfile(join(processed_dir, self.params.dff_video_name+".mp4")) \
            or self.params.overwrite:
            if not isinstance(mask, np.ndarray):
                if (isinstance(mask, bool) and mask is True) \
                    or mask == "fromfile" or mask == "compute":
                    mask_dir = join(processed_dir, self.params.dff_mask)
                    if not os.path.isfile(mask_dir) or self.params.overwrite:
                        dff_stack = get_stack(join(processed_dir, self.params.dff))
                        dff_baseline = get_stack(join(processed_dir, self.params.dff_baseline))
                        if dff_baseline.shape != dff_stack.shape[1:]:
                            crop = (np.array(dff_baseline.shape) - \
                                    np.array(dff_stack.shape[1:])) // 2
                        else:
                            crop = None
                        mask = dff.find_dff_mask(join(processed_dir, self.params.dff_baseline),
                                                 crop=crop)
                        save_stack(mask_dir, mask)
                    else:
                        mask = get_stack(mask_dir)
                        mask = mask > 0

            make_video_dff(dff=join(processed_dir, self.params.dff),
                            out_dir=processed_dir,
                            video_name=self.params.dff_video_name,
                            trial_dir=trial_dir,
                            vmin=self.params.dff_video_vmin,
                            vmax=self.params.dff_video_vmax,
                            pmin=self.params.dff_video_pmin,
                            pmax=self.params.dff_video_pmax,
                            blur=0, mask=mask, crop=None,
                            text=trial_name)

    def _make_dff_videos(self, mask=None):
        """Make one individual dff video for every trial and one merged video for all of them.
        Uses the following params:
            overwrite
            dff_mask
            dff
            dff_baseline
            dff_video_name
            dff_video_vmin
            dff_video_vmax
            dff_video_pmin
            dff_video_pmax

        Parameters
        ----------
        mask : bool or np.array or str, optional
            if numpy array, use it directly as mask to make vide.
            if "fromfile", load the mask specified at fly_processed_dir/self.params.dff_mask
            if bool and True, automatically find a mask (NOT RECOMMENDED!!!),
            by default None
        """
        if not isinstance(mask, np.ndarray):
            if (isinstance(mask, bool) and mask is True) \
                or mask == "fromfile" or mask == "compute":
                mask_dir = join(self.fly_processed_dir, self.params.dff_mask)
                if not os.path.isfile(mask_dir) or self.params.overwrite:
                    dff_stack = get_stack(join(self.trial_processed_dirs[0], self.params.dff))
                    dff_baseline = get_stack(join(self.fly_processed_dir, self.params.dff_baseline))
                    if dff_baseline.shape != dff_stack.shape[1:]:
                        crop = (np.array(dff_baseline.shape) - np.array(dff_stack.shape[1:])) // 2
                        del dff_stack, dff_baseline
                    else:
                        crop = None
                    mask = dff.find_dff_mask(
                        join(self.fly_processed_dir, self.params.dff_baseline),
                        crop=crop)
                    save_stack(mask_dir, mask)
                else:
                    mask = get_stack(mask_dir)
        for i_trial, _ in enumerate(self.trial_dirs):
            self._make_dff_video_trial(i_trial, mask=mask)
        if not os.path.isfile(join(self.fly_processed_dir,
                                   self.params.dff_video_name+"_multiple.mp4")) \
            or self.params.overwrite:
            make_multiple_video_dff(dffs=[join(processed_dir, self.params.dff)
                                        for processed_dir in self.trial_processed_dirs],
                                    out_dir=self.fly_processed_dir,
                                    video_name=self.params.dff_video_name+"_multiple",
                                    trial_dir=self.trial_dirs[0],
                                    vmin=self.params.dff_video_vmin,
                                    vmax=self.params.dff_video_vmax,
                                    pmin=self.params.dff_video_pmin,
                                    pmax=self.params.dff_video_pmax,
                                    share_lim=self.params.dff_video_share_lim,
                                    share_mask=True,
                                    blur=0, mask=mask, crop=None,
                                    text=self.trial_names)

    def _make_dff_behaviour_video_trial(self, i_trial, mask=None, include_2p=False):
        """make combined dff + behaviour (+ raw data) video for one trial.
        Uses the following params:
            green_com_warped
            red_com_warped
            overwrite
            default_video_camera
            dff_beh_video_name
            dff_mask
            dff
            dff_baseline
            dff_video_name
            dff_video_vmin
            dff_video_vmax
            dff_video_pmin
            dff_video_pmax
        Parameters
        ----------
        i_trial : int
            which trial to make the video for

        mask : bool or np.array or str, optional
            if numpy array, use it directly as mask to make vide.
            if "fromfile", load the mask specified at fly_processed_dir/self.params.dff_mask
            if bool and True, automatically find a mask (NOT RECOMMENDED!!!),
            by default None

        include_2p : bool, optional
            whether to also show motion corrected red&green images or just dff, by default False
        """
        processed_dir = self.trial_processed_dirs[i_trial]
        trial_dir = self.trial_dirs[i_trial]
        beh_trial_dir = self.beh_trial_dirs[i_trial]
        sync_trial_dir = self.sync_trial_dirs[i_trial]
        trial_name = self.trial_names[i_trial]
        green_dir = join(self.trial_processed_dirs[i_trial], self.params.green_com_warped) \
            if include_2p else None
        red_dir = join(self.trial_processed_dirs[i_trial], self.params.red_com_warped) \
            if include_2p else None

        if not os.path.isfile(join(processed_dir, self.params.dff_beh_video_name+".mp4")) \
            or self.params.overwrite:
            if not isinstance(mask, np.ndarray):
                if (isinstance(mask, bool) and mask is True) \
                    or mask == "fromfile" or mask == "compute":
                    mask_dir = join(processed_dir, self.params.dff_mask)
                    if not os.path.isfile(mask_dir) or self.params.overwrite:
                        dff_stack = get_stack(join(processed_dir, self.params.dff))
                        dff_baseline = get_stack(join(processed_dir, self.params.dff_baseline))
                        if dff_baseline.shape != dff_stack.shape[1:]:
                            crop = (np.array(dff_baseline.shape) - \
                                    np.array(dff_stack.shape[1:])) // 2
                        else:
                            crop = None
                        mask = dff.find_dff_mask(join(processed_dir, self.params.dff_baseline),
                                                 crop=crop)
                        save_stack(mask_dir, mask)
                    else:
                        mask = get_stack(mask_dir)

            make_video_raw_dff_beh(dff=join(processed_dir, self.params.dff),
                            trial_dir=trial_dir,
                            out_dir=processed_dir,
                            video_name=self.params.dff_beh_video_name,
                            beh_dir=beh_trial_dir,
                            sync_dir=sync_trial_dir,
                            camera=self.params.default_video_camera,
                            stack_axis=0,
                            green=green_dir,
                            red=red_dir,
                            vmin=self.params.dff_video_vmin,
                            vmax=self.params.dff_video_vmax,
                            pmin=self.params.dff_video_pmin,
                            pmax=self.params.dff_video_pmax,
                            blur=0, mask=mask, crop=None, text=trial_name,
                            asgenerator=False, downsample=10)

    def _make_dff_behaviour_video_multiple_trials(self, i_trials=None, mask=None, include_2p=False,
                                                  select_frames=None):
        """make a combined dff + behaviour (+ motion corrected data) video
        for a number of selected trials.
        Uses the following params:
            dff_mask
            overwrite
            dff
            dff_baseline
            green_com_warped
            red_com_warped
            dff_beh_video_name
            default_video_camera
            dff_video_share_lim
            dff_video_log_lim
            dff_video_downsample
            dff_video_vmin
            dff_video_vmax
            dff_video_pmin
            dff_video_pmax

        Parameters
        ----------
        i_trials : list of int, optional
            indices of trials in self.trial_dirs to be used for the video.
            if None, use all trials, by default None

        mask : bool or np.array or str, optional
            if numpy array, use it directly as mask to make vide.
            if "fromfile", load the mask specified at fly_processed_dir/self.params.dff_mask
            if bool and True, automatically find a mask (NOT RECOMMENDED!!!),
            by default None

        include_2p : bool, optional
            whether to also show motion corrected red&green images or just dff, by default False

        select_frames : list of (list or numpy array), optional
            for each trial, list containing a sequence of selected two-photon frame inidices.
            if an empty list, generate a black frame. If None, use all frames, by default None
        """
        if not isinstance(mask, np.ndarray):
            if (isinstance(mask, bool) and mask is True) \
                or mask == "fromfile" or mask == "compute":
                mask_dir = join(self.fly_processed_dir, self.params.dff_mask)
                if not os.path.isfile(mask_dir) or self.params.overwrite:
                    dff_stack = get_stack(join(self.trial_processed_dirs[0], self.params.dff))
                    dff_baseline = get_stack(join(self.fly_processed_dir, self.params.dff_baseline))
                    if dff_baseline.shape != dff_stack.shape[1:]:
                        crop = (np.array(dff_baseline.shape) - np.array(dff_stack.shape[1:])) // 2
                        del dff_stack, dff_baseline
                    else:
                        crop = None
                    mask = dff.find_dff_mask(
                        join(self.fly_processed_dir, self.params.dff_baseline),
                        crop=crop)
                    save_stack(mask_dir, mask)
                else:
                    mask = get_stack(mask_dir)
        # for i_trial, _ in enumerate(self.trial_dirs):
        #     self._make_dff_behaviour_video_trial(i_trial, mask=mask)
        if i_trials is None:
            i_trials = range(len(self.trial_processed_dirs))
        dffs = [join(self.trial_processed_dirs[i_trial], self.params.dff) for i_trial in i_trials]
        trial_dirs = [self.trial_dirs[i_trial] for i_trial in i_trials]
        beh_dirs = [self.beh_trial_dirs[i_trial] for i_trial in i_trials]
        sync_dirs = [self.sync_trial_dirs[i_trial] for i_trial in i_trials]
        text = [self.trial_names[i_trial] for i_trial in i_trials]
        green_dirs = [join(self.trial_processed_dirs[i_trial], self.params.green_com_warped)
                      for i_trial in i_trials] if include_2p else None
        red_dirs = [join(self.trial_processed_dirs[i_trial], self.params.red_com_warped)
                    for i_trial in i_trials] if include_2p else None

        if not os.path.isfile(join(self.fly_processed_dir,
                                   self.params.dff_beh_video_name+"_multiple.mp4")) \
            or self.params.overwrite:
            make_multiple_video_raw_dff_beh(dffs=dffs,
                                        trial_dirs=trial_dirs,
                                        out_dir=self.fly_processed_dir,
                                        video_name=self.params.dff_beh_video_name+"_multiple",
                                        beh_dirs=beh_dirs,
                                        sync_dirs=sync_dirs,
                                        camera=self.params.default_video_camera,
                                        stack_axes=[0, 1],
                                        greens=green_dirs,
                                        reds=red_dirs,
                                        vmin=self.params.dff_video_vmin,
                                        vmax=self.params.dff_video_vmax,
                                        pmin=self.params.dff_video_pmin,
                                        pmax=self.params.dff_video_pmax,
                                        share_lim=self.params.dff_video_share_lim,
                                        log_lim=self.params.dff_video_log_lim,
                                        share_mask=True,
                                        blur=0, mask=mask, crop=None,
                                        text=text,
                                        select_frames=select_frames,
                                        downsample=self.params.dff_video_downsample)  # 10)

    def _compute_summary_stats(self, i_trials=None, raw_only=False, force_overwrite=False):
        """compute summary statistics for every trial and save them for quick access.
        These statistics include the mean/standard deviation/maximum
        of the dff, green denoised, green raw.
        Saves them to a pickle file in the fly_processed_dir.
        Uses the following parameters:
            summary_stats
            overwrite
            dff
            green_denoised
            green_com_warped
        Parameters
        ----------
        i_trials : list of int, optional
            Which trials to compute from. If not supplied, use all. by default None

        raw_only : bool, optional
            Compute only for green raw data. Might be usefull before processing. by default False

        force_overwrite : bool, optional
            overturns the self.params.overwrite flag to force overwriting existing files,
            by default False
        """
        output = join(self.fly_processed_dir, self.params.summary_stats)
        if os.path.isfile(output) and (not self.params.overwrite or force_overwrite):
            return

        if i_trials is None:
            i_trials = range(len(self.trial_processed_dirs))

        dffs = [join(self.trial_processed_dirs[i_trial], self.params.dff) for i_trial in i_trials]
        greens = [join(self.trial_processed_dirs[i_trial], self.params.green_denoised)
                  for i_trial in i_trials]
        greens_raw = [join(self.trial_processed_dirs[i_trial], self.params.green_com_warped)
                  for i_trial in i_trials]

        if not raw_only:
            good_dffs = []
            bad_dffs = []
            for i_dff, dff_dir in enumerate(dffs):
                try:
                    dff = get_stack(dff_dir)
                    good_dffs.append(i_dff)
                except FileNotFoundError:
                    print("could not find: "+dff_dir+" \nWill replace with zeros." )
                    dff = None
                    bad_dffs.append(i_dff)
                dffs[i_dff] = dff
            for i_bad in bad_dffs:
                dffs[i_bad] = np.zeros_like(dffs[good_dffs[0]])

            # compute quantities
            means = [np.mean(dff, axis=0) for dff in dffs]
            mean_diffs = [mean - means[good_dffs[0]] for mean in means]
            stds = [np.std(dff, axis=0) for dff in dffs]
            std_diffs = [std - stds[good_dffs[0]] for std in stds]
            quants = [np.percentile(dff, 95, axis=0) for dff in dffs]
            quant_diffs = [quant - quants[good_dffs[0]] for quant in quants]
            local_corrs = [local_correlations(dff) for dff in dffs]
            local_corr_diffs = [local_corr - local_corrs[good_dffs[0]]
                                for local_corr in local_corrs]

        del dffs
        if not raw_only:
            # get green stacks
            greens = [get_stack(green) for green in greens]

            # compute quantities
            means_green = [np.mean(green, axis=0) for green in greens]
            mean_diffs_green = [mean - means_green[0] for mean in means_green]
            stds_green = [np.std(green, axis=0) for green in greens]
            std_diffs_green = [std - stds_green[0] for std in stds_green]
            quants_green = [np.percentile(green, 95, axis=0) for green in greens]
            quant_diffs_green = [quant - quants_green[0] for quant in quants_green]
            local_corrs_green = [local_correlations(green) for green in greens]
            local_corr_diffs_green = [local_corr - local_corrs_green[0]
                                    for local_corr in local_corrs_green]

        del greens

        # get green stacks
        greens_raw = [get_stack(green) for green in greens_raw]

        # compute quantities
        means_green_raw = [np.mean(green, axis=0) for green in greens_raw]
        mean_diffs_green_raw = [mean - means_green_raw[0] for mean in means_green_raw]
        stds_green_raw = [np.std(green, axis=0) for green in greens_raw]
        std_diffs_green_raw = [std - stds_green_raw[0] for std in stds_green_raw]
        quants_green_raw = [np.percentile(green, 95, axis=0) for green in greens_raw]
        quant_diffs_green_raw = [quant - quants_green_raw[0] for quant in quants_green_raw]
        local_corrs_green_raw = [local_correlations(green) for green in greens_raw]
        local_corr_diffs_green_raw = [local_corr - local_corrs_green_raw[0]
                                  for local_corr in local_corrs_green_raw]

        del greens_raw

        # make dictionary
        if not raw_only:
            output_dict = {
                "dff_means": means,
                "dff_mean_diffs": mean_diffs,
                "green_means": means_green,
                "green_mean_diffs": mean_diffs_green,
                "green_means_raw": means_green_raw,
                "green_mean_diffs_raw": mean_diffs_green_raw,
                "dff_stds": stds,
                "dff_std_diffs": std_diffs,
                "green_stds": stds_green,
                "green_std_diffs": std_diffs_green,
                "green_stds_raw": stds_green_raw,
                "green_std_diffs_raw": std_diffs_green_raw,
                "dff_quants": quants,
                "dff_quant_diffs": quant_diffs,
                "green_quants": quants_green,
                "green_quant_diffs": quant_diffs_green,
                "green_quants_raw": quants_green_raw,
                "green_quant_diffs_raw": quant_diffs_green_raw,
                "dff_local_corrs": local_corrs,
                "dff_local_corr_diffs": local_corr_diffs,
                "green_local_corrs": local_corrs_green,
                "green_local_corr_diffs": local_corr_diffs_green,
                "green_local_corrs_raw": local_corrs_green_raw,
                "green_local_corr_diffs_raw": local_corr_diffs_green_raw,
                "trials": self.trial_names,
                "ref_trial": good_dffs[0]
            }
        else:
            output_dict = {
                "green_means_raw": means_green_raw,
                "green_mean_diffs_raw": mean_diffs_green_raw,
                "green_stds_raw": stds_green_raw,
                "green_std_diffs_raw": std_diffs_green_raw,
                "green_quants_raw": quants_green_raw,
                "green_quant_diffs_raw": quant_diffs_green_raw,
                "green_local_corrs_raw": local_corrs_green_raw,
                "green_local_corr_diffs_raw": local_corr_diffs_green_raw,
                "trials": self.trial_names,
            }

        with open(output, "wb") as f:
            pickle.dump(output_dict, f)

    def get_dfs(self):
        """Get synchronised data frames including the frame times for two photon
        and behavioural data.
        Get as much behavioural data into these data frames as possible.
        This includes: optical flow or fictrac data, pose estimation data
        Uses the following params:
            opflow_df_out_dir
            df3d_df_out_dir
            twop_df_out_dir
            denoise_params
            ball_tracking
            opflow_win_size
            thres_rest
            thres_walk
            add_df3d_to_df

        Raises
        ------
        KeyError
            if keyboard interrupt during processing is sent, otherwise ignores errors and continues
        """
        for i_trial, (trial_dir, processed_dir, trial_name) \
            in enumerate(zip(self.trial_dirs, self.trial_processed_dirs, self.trial_names)):
            print(time.ctime(time.time()), " creating data frames: " + trial_dir)

            trial_info = {"Date": self.date,
                        "Genotype": self.genotpye,
                        "Fly": self.fly,
                        "TrialName": trial_name,
                        "i_trial": self.selected_trials[i_trial]
                        }
            if not os.path.isdir(processed_dir):
                os.makedirs(processed_dir)
            opflow_out_dir = os.path.join(processed_dir, self.params.opflow_df_out_dir)
            df3d_out_dir = os.path.join(processed_dir, self.params.df3d_df_out_dir)
            twop_out_dir = os.path.join(processed_dir, self.params.twop_df_out_dir)
            # if not os.path.isfile(opflow_out_dir) or not os.path.isfile(df3d_out_dir) \
            #     or not os.path.isfile(twop_out_dir) or self.params.overwrite:
            try:
                _1, _2, _3 = get_synchronised_trial_dataframes(
                    trial_dir,
                    crop_2p_start_end=self.params.denoise_params.pre_post_frame,
                    beh_trial_dir=self.beh_trial_dirs[i_trial],
                    sync_trial_dir=self.sync_trial_dirs[i_trial],
                    trial_info=trial_info,
                    opflow=True if self.params.ball_tracking == "opflow" else False,
                    df3d=True,
                    opflow_out_dir=opflow_out_dir,
                    df3d_out_dir=df3d_out_dir,
                    twop_out_dir=twop_out_dir
                )
                if self.params.ball_tracking == "opflow":
                    opflow_df, frac_walk_rest = get_opflow_df(self.beh_trial_dirs[i_trial],
                                                            index_df=opflow_out_dir,
                                                            df_out_dir=opflow_out_dir,
                                                            block_error=True,
                                                            return_walk_rest=True,
                                                            winsize=self.params.opflow_win_size,
                                                            thres_rest=self.params.thres_rest,
                                                            thres_walk=self.params.thres_walk)
                    _ = get_opflow_in_twop_df(twop_df=twop_out_dir,
                                              opflow_df=opflow_df,
                                              twop_df_out_dir=twop_out_dir,
                                              thres_walk=self.params.thres_walk,
                                              thres_rest=self.params.thres_rest)
                    print("walking, resting: ", frac_walk_rest)
                elif self.params.ball_tracking == "fictrac":
                    _ = get_fictrac_df(self.beh_trial_dirs[i_trial],
                                       index_df=df3d_out_dir,
                                       df_out_dir=df3d_out_dir)

                if self.params.add_df3d_to_df:
                    _ = df3d.get_df3d_dataframe(self.beh_trial_dirs[i_trial],
                                                index_df=df3d_out_dir,
                                                out_dir=df3d_out_dir)
            except KeyboardInterrupt:
                raise KeyError
            except:
                print("Error while getting dfs in trial: " + trial_dir)

    def extract_rois(self):
        """extract the regions of interest from manually labelled ROI centers.
        Uses the following params:
            roi_centers
            roi_mask
            twop_df_out_dir
            green_denoised
            roi_size
            roi_pattern
        """
        roi_file = os.path.join(self.fly_processed_dir, self.params.roi_centers)
        mask_out_dir = os.path.join(self.fly_processed_dir, self.params.roi_mask)
        for processed_dir in self.trial_processed_dirs:
            print(time.ctime(time.time()), " extracting ROIs: " + processed_dir)
            twop_out_dir = os.path.join(processed_dir, self.params.twop_df_out_dir)
            stack = os.path.join(processed_dir, self.params.green_denoised)
            _ = get_roi_signals_df(stack, roi_file,
                                    size=self.params.roi_size, pattern=self.params.roi_pattern,
                                    index_df=twop_out_dir, df_out_dir=twop_out_dir,
                                    mask_out_dir=mask_out_dir)
