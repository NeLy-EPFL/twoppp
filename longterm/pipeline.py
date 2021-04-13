# Jonas Braun
# jonas.braun@epfl.ch
# 17.03.2021

import os, sys
from os.path import join
import gc
import numpy as np
from copy import deepcopy

from ofco.utils import default_parameters
from deepinterpolation import interface as denoise

FILE_PATH = os.path.realpath(__file__)
LONGTERM_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, dff
from longterm.utils import makedirs_safe, get_stack
from longterm.register import warping
from longterm.behaviour import df3d

class PreProcessParams:
    def __init__(self):
        # names of tif/npy files generated throughout processing
        self.red_raw = "red.tif"
        self.green_raw = "green.tif"
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

        self.ref_frame = "ref_frame_com.tif"
        self.com_offset = "com_offset.npy"
        self.motion_field = "motion_field_com.npy"

        # mode of pre-processing: if True, perfom one processing step 
        # on each trial before moving to next processing step
        self.breadth_first = True
        self.overwrite = False
        self.use_warp = True
        self.use_denoise = True
        self.use_dff = True
        self.use_df3d = True
        self.use_df3dPostProcess = True
        self.use_behaviour_classifier = False
        self.select_trials = False

        # ofco params
        self.i_ref_trial = 0
        self.i_ref_frame = 0
        self.use_com = True
        self.save_motion_field = False  # TODO: set this to True if we have enought space
        self.ofco_param = default_parameters()
        self.ofco_parallel = True
        # self.ofco_out_dtype = np.float32  # TODO: verify whether this is the best choice

        # denoising params
        self.denoise_crop_size = (320, 640)
        self.denoise_crop_offset = (None, None)
        self.denoise_tmp_data_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation", "data")
        self.denoise_tmp_data_name = self._make_denoise_data_name
        self.denoise_tmp_run_dir = join(os.path.expanduser("~"), "tmp", "deepinterpolation", "runs")
        self.denoise_runid = self._make_runid
        self.denoise_final_dir = "denoising_run"
        self.denoise_delete_tmp_run_dir = False  # TODO: change this to True
        self.denoise_params = denoise.DefaultInterpolationParams()

        # dff params
        self.dff_common_baseline = True
        self.dff_baseline_blur = 1
        self.dff_baseline_med_filt = 3
        self.dff_baseline_blur_pre = False
        self.dff_baseline_mode = "convolve"
        self.dff_baseline_length = 10
        self.dff_baseline_quantile = 0.95
        self.dff_use_crop=False
        self.dff_manual_add_to_crop = 20
        self.dff_blur = 0

        # deepfly3d params
        self.behaviour_as_videos = True
        self.twop_scope = 2

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
    def __init__(self, fly_dir, params=PreProcessParams(), beh_fly_dir=None):
        super().__init__()
        self.params = params
        self.fly_dir = fly_dir
        self.trial_dirs = load.get_trials_from_fly(self.fly_dir)[0]
        self.beh_fly_dir = self.fly_dir if beh_fly_dir is None else beh_fly_dir
        self.beh_trial_dirs = load.get_trials_from_fly(self.beh_fly_dir)[0]
        
        self.match_trials_and_beh_trials()
        self.create_processed_structure()
        self.save_ref_frame()

    def match_trials_and_beh_trials(self):
        if len(self.trial_dirs) == len(self.beh_trial_dirs) \
            and all([trial_dir == beh_trial_dir 
                     for trial_dir, beh_trial_dir 
                     in zip(self.trial_dirs, self.beh_trial_dirs)]):
            return
        else:
            new_beh_trial_dirs = []
            new_trial_dirs = deepcopy(self.trial_dirs)
            for trial_dir in self.trial_dirs:
                # match behaviour trial to all existing 2p trials
                _, trial = os.path.split(trial_dir)
                beh_trial_ends = [beh_trial_dir
                                    for beh_trial_dir in self.beh_trial_dirs
                                    if beh_trial_dir.endswith(trial)]
                if len(beh_trial_ends) == 1:
                    new_beh_trial_dirs.append(beh_trial_ends[0])
                elif len(beh_trial_ends) == 0:
                    new_beh_trial_dirs.append("")
                    print("Did not find matching behaviour trial for " + trial_dir)
                else:
                    raise(AssertionError, "found multiple behavioural trials ending with " + trial)
            for beh_trial_dir in self.beh_trial_dirs:
                # add remaining behaviour trials that were not matched to 2p trials
                if beh_trial_dir not in new_beh_trial_dirs:
                    new_beh_trial_dirs.append(beh_trial_dir)
                    new_trial_dirs.append("")
                    print("Did not find matching 2p trial for " + beh_trial_dir)
            self.trial_dirs = new_trial_dirs
            self.beh_trial_dirs = new_beh_trial_dirs

    def create_processed_structure(self):
        self.fly_processed_dir = join(self.fly_dir, load.PROCESSED_FOLDER)
        makedirs_safe(self.fly_processed_dir)
        self.trial_processed_dirs = [join(trial_dir, load.PROCESSED_FOLDER) if trial_dir != "" else ""
                                     for trial_dir in self.trial_dirs]
        _ = [makedirs_safe(processed_dir)
             for processed_dir in self.trial_processed_dirs if processed_dir != ""]

    def save_ref_frame(self):
        ref_trial_dir = self.trial_dirs[self.params.i_ref_trial]
        ref_processed_dir = self.trial_processed_dirs[self.params.i_ref_trial]
        _ = load.convert_raw_to_tiff(ref_trial_dir, 
                                     overwrite=self.params.overwrite, 
                                     return_stacks=False)
        ref_stack = join(ref_processed_dir, self.params.red_raw)
        self.ref_frame = join(self.fly_processed_dir, self.params.ref_frame)
        warping.save_ref_frame(stack=ref_stack,
                               ref_frame_dir=self.ref_frame,
                               i_frame=self.params.i_ref_frame,
                               com_pre_reg=self.params.use_com,
                               overwrite=self.params.overwrite)

    def run_all_trials(self):
        if self.params.breadth_first:
            self._run_breadth_first()
        else:
            self._run_depth_first()

    def run_single_trial(self, i_trial):
        trial_dir = self.trial_dirs[i_trial]
        beh_trial_dir = self.beh_trial_dirs[i_trial]
        processed_dir = self.trial_processed_dirs[i_trial]
        self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
        if self.params.use_warp:
            self._warp_trial(processed_dir)
        if self.params.use_denoise:
            self._denoise_trial(processed_dir)
        if self.params.use_dff:
            self._compute_dff_trial(processed_dir, force_single_baseline=True)
        if self.params.use_df3d:
            self._pose_estimate(beh_trial_dir)
        if self.params.use_df3dPostProcess:
            self._post_process_pose(beh_trial_dir)

    def _run_breadth_first(self):
        for i_trial, (trial_dir, processed_dir) in \
            enumerate(zip(self.trial_dirs, self.trial_processed_dirs)):
            print("===== converting trial to tif: " + trial_dir)
            self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
            gc.collect()
        if self.params.use_warp:
            _ = [self._warp_trial(processed_dir) 
                 for processed_dir in self.trial_processed_dirs]
            gc.collect()
        if self.params.use_denoise:
            _ = [self._denoise_trial(processed_dir) 
                 for processed_dir in self.trial_processed_dirs]
        if self.params.use_dff:
            print("===== computing dff")
            self._compute_dff_alltrials()
        if self.params.use_df3d:
            print("===== performing pose estimation with deepfly3d")
            self._pose_estimate()
        if self.params.use_df3dPostProcess:
            print("===== post processing df3d pose")
            self._post_process_pose()
        
    def _run_depth_first(self, force_single_trial_dff=False):
        if isinstance(self.params.select_trials, bool) and not self.params.select_trials:
            selected_trials = [True for trial in self.trial_dirs]
        elif isinstance(self.params.select_trials, list) and len(self.params.select_trials) == len(self.trial_dirs):
            selected_trials = self.params.select_trials
        else:
            raise NotImplementedError
        for i_trial, (trial_dir, beh_trial_dir, processed_dir) in \
            enumerate(zip(self.trial_dirs, self.beh_trial_dirs, self.trial_processed_dirs)):
            if selected_trials[i_trial]:
                self._convert_raw_to_tiff_trial(trial_dir, processed_dir)
                if self.params.use_warp:
                    self._warp_trial(processed_dir)
                if self.params.use_denoise:
                    self._denoise_trial(processed_dir)
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
        
    def _convert_raw_to_tiff_trial(self, trial_dir, processed_dir):
        if trial_dir != "" and processed_dir != "":
            _ = load.convert_raw_to_tiff(trial_dir, 
                                        overwrite=self.params.overwrite, 
                                        return_stacks=False,
                                        green_dir=join(processed_dir, self.params.green_raw),
                                        red_dir=join(processed_dir, self.params.red_raw)
                                        )
    
    def _warp_trial(self, processed_dir):
        if processed_dir != "":
            print("===== warping trial: " + processed_dir)
            _ = warping.warp(stack1=join(processed_dir, self.params.red_raw),
                            stack2=join(processed_dir, self.params.green_raw),
                            ref_frame=self.ref_frame,
                            stack1_out_dir=join(processed_dir, self.params.red_com_warped),
                            stack2_out_dir=join(processed_dir, self.params.green_com_warped),
                            com_pre_reg=self.params.use_com,
                            offset_dir=join(processed_dir, self.params.com_offset),
                            return_stacks=False,
                            overwrite=self.params.overwrite,
                            select_frames=None,
                            parallel=self.params.ofco_parallel,
                            verbose=False,
                            w_output=join(processed_dir, self.params.motion_field)
                            if self.params.save_motion_field else None,
                            initial_w=None,
                            param=self.params.ofco_param
                            )

    def _denoise_trial(self, processed_dir):
        if processed_dir != "":
            print("===== denoising trial: " + processed_dir)
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
            # TODO:
            # denoise.copy_run_dir(tmp_run_dir,
            #                      join(processed_dir, self.params.denoise_final_dir),
            #                      delete_tmp=self.params.denoise_delete_tmp_run_dir)
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
                                                dff_out_dir=join(processed_dir, self.params.dff),
                                                return_stack=False)

    def _compute_dff_alltrials(self):
        stacks = [join(processed_dir, self.params.green_denoised) 
                  for processed_dir in self.trial_processed_dirs if processed_dir != ""]
        baseline_dirs = [join(processed_dir, self.params.dff_baseline)
                         for processed_dir in self.trial_processed_dirs if processed_dir != ""]
        if self.params.dff_common_baseline:
            print("===== computing dff baseline")
            if not os.path.isfile(join(self.fly_processed_dir, self.params.dff_baseline)) or self.params.overwrite:
                _ = dff.find_dff_baseline_multi_stack_load_single(stacks=stacks,
                        individual_baselin_dirs=baseline_dirs,
                        baseline_blur=self.params.dff_baseline_blur,
                        baseline_med_filt=self.params.dff_baseline_med_filt,
                        blur_pre=self.params.dff_baseline_blur_pre,
                        baseline_mode=self.params.dff_baseline_mode,
                        baseline_length=self.params.dff_baseline_length,
                        baseline_quantile=self.params.dff_baseline_quantile,
                        baseline_dir=join(self.fly_processed_dir, self.params.dff_baseline))
               
        for i_trial, processed_dir \
            in enumerate(self.trial_processed_dirs):
            print("===== computing dff for trial: " + processed_dir)
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
                print("===== postprocessing df3d for trial: " + trial_dir)
                df3d.postprocess_df3d_trial(trial_dir, overwrite=self.params.overwrite)         



if __name__ == "__main__":
    date_dir = os.path.join(load.NAS_DIR_JB, "210301_J1xCI9")
    fly_dirs = load.get_flies_from_datedir(date_dir=date_dir)
    fly_dir = fly_dirs[0]

    params = PreProcessParams()
    preprocess = PreProcessFly(fly_dir, params=params)

    preprocess.run_all_trials()