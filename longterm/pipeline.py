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

FILE_PATH = os.path.realpath(__file__)
LONGTERM_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)
OUTPUT_PATH = os.path.join(MODULE_PATH, "outputs")

from longterm import load, dff
from longterm.utils import makedirs_safe, get_stack, save_stack, readlines_tolist
from longterm.register import warping
from longterm.denoise import prepare_corrected_data
from longterm.behaviour import df3d
from longterm.plot.videos import make_video_dff, make_multiple_video_dff, make_video_raw_dff_beh, make_multiple_video_raw_dff_beh
from longterm.rois import local_correlations, get_roi_signals_df
from longterm.behaviour.synchronisation import get_synchronised_trial_dataframes
from longterm.behaviour.optic_flow import get_opflow_df, get_opflow_in_twop_df
from longterm.behaviour.fictrac import get_fictrac_df
from longterm.analysis import InterPCAAnalysis

class PreProcessParams:
    """Class containing all default parameters for the PreProcessFly class."""

    def __init__(self):
        """Class containing all default parameters for the PreProcessFly class."""
        self.genotype = " "
        # names of tif/npy files generated throughout processing
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

        self.pca_analysis_file = "pcan.pkl"
        self.pca_analysis_map_file = "pca_map.pdf"

        # mode of pre-processing: if True, perfom one processing step
        # on each trial before moving to next processing step
        self.breadth_first = True
        self.overwrite = False
        self.use_warp = False
        self.use_denoise = False
        self.use_dff = False
        self.use_df3d = False
        self.use_df3dPostProcess = False
        self.use_behaviour_classifier = False
        self.select_trials = False
        self.cleanup_files = False
        self.make_dff_videos = False
        self.make_summary_stats = True
        self.ball_tracking = "opflow"  # "opflow", "fictrac", or None

        # ofco params
        self.i_ref_trial = 0
        self.i_ref_frame = 0
        self.use_com = True
        self.post_com_crop = True
        self.post_com_crop_values = None
        self.save_motion_field = False  # TODO: set this to True if we have enought space
        self.ofco_param = default_parameters()
        self.ofco_parallel = True
        self.ofco_verbose = True
        # self.ofco_out_dtype = np.float32  # TODO: verify whether this is the best choice

        # denoising params
        self.denoise_crop_size = (320, 640)
        self.denoise_crop_offset = (None, None)
        self.denoise_tmp_data_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation", "data")
        self.denoise_tmp_data_name = self._make_denoise_data_name
        self.denoise_tmp_run_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation", "runs")
        self.denoise_runid = self._make_runid
        self.denoise_final_dir = "denoising_run"
        self.denoise_delete_tmp_run_dir = True
        self.denoise_params = denoise.DefaultInterpolationParams()
        self.denoise_train_each_trial = False
        self.denoise_train_trial = 0
        self.denoise_correct_illumination_leftright = False

        # dff params
        self.dff_common_baseline = True
        self.dff_baseline_blur = 10
        self.dff_baseline_med_filt = 1
        self.dff_baseline_blur_pre = True
        self.dff_baseline_mode = "convolve"
        self.dff_baseline_length = 10
        self.dff_baseline_quantile = 0.95
        self.dff_use_crop=False
        self.dff_manual_add_to_crop = 20
        self.dff_blur = 0
        self.dff_min_baseline = None
        self.dff_baseline_exclude_trials = None

        self.dff_video_name = "dff"
        self.dff_beh_video_name = "dff_beh"
        self.dff_video_pmin = None
        self.dff_video_vmin = 0
        self.dff_video_pmax = 99
        self.dff_video_vmax = None
        self.dff_video_share_lim = True
        self.dff_video_log_lim = False
        self.dff_video_downsample = 2
        self.default_video_camera = 6

        # deepfly3d params
        self.behaviour_as_videos = True
        self.twop_scope = 2

        # optic flow params
        self.opflow_win_size = 80
        self.thres_walk = 0.03
        self.thres_rest = 0.01

        # ROI extraction params
        self.roi_size = (7,11)
        self.roi_pattern = "default"



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
                 beh_fly_dir=None, beh_trial_dirs=None, sync_fly_dir=None, sync_trial_dirs=None):
        super().__init__()
        self.params = params
        self.fly_dir = fly_dir
        self.selected_trials = selected_trials
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
        self.beh_fly_dir, _ = os.path.split(self.beh_trial_dirs[0]) if beh_fly_dir is None else beh_fly_dir

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
        self.sync_fly_dir, _ = os.path.split(self.sync_trial_dirs[0]) if sync_fly_dir is None else sync_fly_dir

        self._match_trials_and_beh_trials()
        if self.selected_trials is not None and isinstance(self.selected_trials, list):
            self.trial_dirs = [self.trial_dirs[i] for i in self.selected_trials]
            self.beh_trial_dirs = [self.beh_trial_dirs[i] for i in self.selected_trials]
            self.sync_trial_dirs = [self.sync_trial_dirs[i] for i in self.selected_trials]
            self._match_trials_and_beh_trials()
        elif self.selected_trials is not None:
            raise NotImplementedError("selected_trials should be None or a list containing integer indices.")
        else:
            self.selected_trials = np.arange(len(self.trial_dirs))
        self._get_trial_names()
        self._get_experiment_info()
        self._create_processed_structure()
        # self._save_ref_frame() --> moved to run_all_trials

    def _get_trial_names(self):
        trial_names = []
        for trial in self.trial_dirs:
            _, name = os.path.split(trial)
            trial_names.append(name)
        self.trial_names = trial_names

    def _get_experiment_info(self):
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
        assert len(self.beh_trial_dirs) == len(self.trial_dirs)
        assert len(self.sync_trial_dirs) == len(self.trial_dirs)
        return

    def _create_processed_structure(self):
        self.fly_processed_dir = join(self.fly_dir, load.PROCESSED_FOLDER)
        makedirs_safe(self.fly_processed_dir)
        self.trial_processed_dirs = [join(trial_dir, load.PROCESSED_FOLDER) if trial_dir != "" else ""
                                     for trial_dir in self.trial_dirs]
        _ = [makedirs_safe(processed_dir)
             for processed_dir in self.trial_processed_dirs if processed_dir != ""]

    def _save_ref_frame(self):
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
        self._save_ref_frame()
        if self.params.breadth_first:
            self._run_breadth_first()
        else:
            self._run_depth_first()

    def run_single_trial(self, i_trial):
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

    def _run_depth_first(self, force_single_trial_dff=False):
        # TODO: align this trial selection with the new one written up in __init__()
        if isinstance(self.params.select_trials, bool) and not self.params.select_trials:
            selected_trials = [True for trial in self.trial_dirs]
        elif isinstance(self.params.select_trials, list) and len(self.params.select_trials) == len(self.trial_dirs):
            selected_trials = self.params.select_trials
        else:
            raise NotImplementedError("selected trials should be a list of booleans with the same length as self.trial_dirs.")
        for i_trial, (trial_dir, beh_trial_dir, processed_dir) in \
            enumerate(zip(self.trial_dirs, self.beh_trial_dirs, self.trial_processed_dirs)):
            if selected_trials[i_trial]:
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

    def _convert_raw_to_tiff_trial(self, trial_dir, processed_dir):
        if trial_dir != "" and processed_dir != "":
            _ = load.convert_raw_to_tiff(trial_dir, 
                                        overwrite=self.params.overwrite, 
                                        return_stacks=False,
                                        green_dir=join(processed_dir, self.params.green_raw),
                                        red_dir=join(processed_dir, self.params.red_raw)
                                        )

    def _com_correct_trial(self, processed_dir):
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
        if processed_dir != "":
            if self.params.use_com and self.params.post_com_crop:
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
        if processed_dir != "":
            print(time.ctime(time.time()), "denoising trial: " + processed_dir)
            if os.path.isfile(join(processed_dir, self.params.green_denoised)) and not self.params.overwrite:
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
        if self.params.denoise_train_each_trial:
            print(time.ctime(time.time()), "Start denoising by training on all trials.")
            _ = [self._denoise_trial_trainNinfer(processed_dir) 
                 for processed_dir in self.trial_processed_dirs]
        else:
            print(time.ctime(time.time()), "Start denoising by training on one trial: {}".format(self.params.denoise_train_trial))
            input_datas = []
            tmp_data_dirs = []
            if os.path.isdir(join(self.fly_processed_dir, self.params.denoise_final_dir)) and not self.params.overwrite:
                tmp_run_dir = join(self.fly_processed_dir, self.params.denoise_final_dir)
                already_trained = True
                print("Using already trained model.")
            else:
                already_trained = False

            already_denoised_trials = [os.path.isfile(join(processed_dir, self.params.green_denoised))
                                       for processed_dir in self.trial_processed_dirs]
            if all(already_denoised_trials) and not self.params.overwrite:
                return
            elif any(already_denoised_trials) and already_trained and not self.params.overwrite:
                todo_trial_processed_dirs = [processed_dir for processed_dir, already_denoised
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
                                       summary_dict_pickle=join(self.fly_processed_dir, self.summary_stats))
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

            if not os.path.isfile(join(processed_dir, self.params.dff)) or self.params.overwrite or force_overwrite:
                _ = dff.compute_dff_from_stack(stack, 
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
        if self.params.dff_baseline_exclude_trials is None:
            self.params.dff_baseline_exclude_trials = [False for _ in self.trial_dirs]
        stacks = [join(processed_dir, self.params.green_denoised) 
                  for i_dir, processed_dir in enumerate(self.trial_processed_dirs) 
                  if processed_dir != "" and not self.params.dff_baseline_exclude_trials[i_dir]]
        baseline_dirs = [join(processed_dir, self.params.dff_baseline)
                         for i_dir, processed_dir in enumerate(self.trial_processed_dirs) 
                         if processed_dir != "" and not self.params.dff_baseline_exclude_trials[i_dir]]
        if self.params.dff_common_baseline:
            print(time.ctime(time.time()), "computing dff baseline")
            if not os.path.isfile(join(self.fly_processed_dir, self.params.dff_baseline)) or self.params.overwrite:
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
        trial_dirs = deepcopy(self.beh_trial_dirs) if trial_dirs is None else trial_dirs
        for trial_dir in trial_dirs:
            if trial_dir != "":
                print(time.ctime(time.time()), "postprocessing df3d for trial: " + trial_dir)
                df3d.postprocess_df3d_trial(trial_dir, overwrite=self.params.overwrite)         

    def _make_dff_video_trial(self, i_trial, mask=None):
        processed_dir = self.trial_processed_dirs[i_trial]
        trial_dir = self.trial_dirs[i_trial]
        trial_name = self.trial_names[i_trial]
        if not os.path.isfile(join(processed_dir, self.params.dff_video_name+".mp4")) \
            or self.params.overwrite:
            if not isinstance(mask, np.ndarray):
                if (isinstance(mask, bool) and mask == True) \
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
        if not isinstance(mask, np.ndarray):
            if (isinstance(mask, bool) and mask == True) \
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
                    mask = dff.find_dff_mask(join(self.fly_processed_dir, self.params.dff_baseline), crop=crop)
                    save_stack(mask_dir, mask)
                else:
                    mask = get_stack(mask_dir)
        for i_trial, _ in enumerate(self.trial_dirs):
            self._make_dff_video_trial(i_trial, mask=mask)
        if not os.path.isfile(join(self.fly_processed_dir, self.params.dff_video_name+"_multiple.mp4")) \
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
        processed_dir = self.trial_processed_dirs[i_trial]
        trial_dir = self.trial_dirs[i_trial]
        beh_trial_dir = self.beh_trial_dirs[i_trial]
        sync_trial_dir = self.sync_trial_dirs[i_trial]
        trial_name = self.trial_names[i_trial]
        green_dir = join(self.trial_processed_dirs[i_trial], self.params.green_com_warped) if include_2p else None
        red_dir = join(self.trial_processed_dirs[i_trial], self.params.red_com_warped) if include_2p else None

        if not os.path.isfile(join(processed_dir, self.params.dff_beh_video_name+".mp4")) \
            or self.params.overwrite:
            if not isinstance(mask, np.ndarray):
                if (isinstance(mask, bool) and mask == True) \
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

    def _make_dff_behaviour_video_multiple_trials(self, i_trials=None, mask=None, include_2p=False, select_frames=None):
        if not isinstance(mask, np.ndarray):
            if (isinstance(mask, bool) and mask == True) \
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
                    mask = dff.find_dff_mask(join(self.fly_processed_dir, self.params.dff_baseline), crop=crop)
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
        green_dirs = [join(self.trial_processed_dirs[i_trial], self.params.green_com_warped) for i_trial in i_trials] if include_2p else None
        red_dirs = [join(self.trial_processed_dirs[i_trial], self.params.red_com_warped) for i_trial in i_trials] if include_2p else None

        if not os.path.isfile(join(self.fly_processed_dir, self.params.dff_beh_video_name+"_multiple.mp4")) \
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
            local_corr_diffs = [local_corr - local_corrs[good_dffs[0]] for local_corr in local_corrs]

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
            except KeyboardInterrupt:
                raise KeyError
            except:
                print("Error while getting dfs and computing optic flow in trial: " + trial_dir)

    def extract_rois(self):
        roi_file = os.path.join(self.fly_processed_dir, "ROI_centers.txt")
        mask_out_dir = os.path.join(self.fly_processed_dir, "ROI_mask.tif")
        for processed_dir in self.trial_processed_dirs:
            print(time.ctime(time.time()), " extracting ROIs: " + processed_dir)
            twop_out_dir = os.path.join(processed_dir, self.params.twop_df_out_dir)
            stack = os.path.join(processed_dir, self.params.green_denoised)
            _ = get_roi_signals_df(stack, roi_file,
                                    size=self.params.roi_size, pattern=self.params.roi_pattern,
                                    index_df=twop_out_dir, df_out_dir=twop_out_dir,
                                    mask_out_dir=mask_out_dir)

    def prepare_pca_analysis(self, condition, i_trials=None, compare_trials=None, load_df=True, load_pixels=True, 
                             pixel_shape=None, sigma=1, zscore_trials="all"):
        print(time.ctime(time.time()), " preparing PCA analysis: " + self.fly_dir)
        i_trials = self.selected_trials if i_trials is None else i_trials
        trial_names = [trial_name for i_trial, trial_name in zip(self.selected_trials, self.trial_names)
                       if i_trial in i_trials]
        pcan = InterPCAAnalysis(fly_dir=self.fly_dir,
                        i_trials=i_trials,
                        condition=condition,
                        compare_i_trials=i_trials if compare_trials is None else compare_trials,
                        thres_walk=self.params.thres_walk,
                        thres_rest=self.params.thres_rest,
                        load_df=load_df,
                        load_pixels=load_pixels,
                        pixel_shape=pixel_shape,
                        sigma=sigma,
                        trial_names=trial_names,
                        zscore_trials=zscore_trials)
        out_file = os.path.join(self.fly_processed_dir, self.params.pca_analysis_map_file)
        print(time.ctime(time.time()), " creating PCA maps: " + self.fly_dir)
        pcan.save_maps(out_file)
        copy_file_name = f"{self.params.pca_analysis_map_file[:-4]}_{self.date}_{self.genotpye}_fly{self.fly}.pdf"
        copy_file = os.path.join(OUTPUT_PATH,copy_file_name)
        copy2(out_file, copy_file)
        out_file = os.path.join(self.fly_processed_dir, self.params.pca_analysis_file)
        print(time.ctime(time.time()), " pickling PCA analysis: " + self.fly_dir)
        pcan.pickle_self(out_file)




if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
    fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)
    fly_dir = fly_dirs[0]

    params = PreProcessParams()
    preprocess = PreProcessFly(fly_dir, params=params)

    preprocess.run_all_trials()