import os, sys
import numpy as np
from copy import deepcopy
from time import time, sleep
from tqdm import tqdm
import gc
import itertools
import time
import pickle
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from glob import glob
import multiprocessing
# import matplotlib
# matplotlib.use('agg')  # use non-interactive backend for PNG plotting

from deepinterpolation import interface as denoise
import tensorflow as tf
MAXMEM = 10*1024+512  #10.5GB
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate a fraction of GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MAXMEM)])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils, dff
from longterm.pipeline import PreProcessFly, PreProcessParams
from longterm.plot.videos import make_2p_grid_video, make_multiple_video_dff

fly_dirs = [
    os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # high caff
    os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
    os.path.join(load.NAS2_DIR_LH, "210811", "fly1"),  # starv
]

all_selected_trials = [
    [1,4,5,7,12],  # [1,3,4,5,6,7,8,9,10,11,12,13],
    [1,4,5,8,12],  # [1,3,4,5,6,8,9,10,11,12],
    [2,4,5,7],
]

denoise_training_trials = [0,2]
denoise_test_trials = [1]

parameter_options = {
    "N_trials": [1, 2],
    "N_frames_per_trial": [1000, 2000],
    "nb_times_through_data": [1,2],
}

dff_parameter_options = {
    "quantile": [False, True],
    "baseline_length": [10,20]
}

# input_tif = "green_com_warped.tif"
# output_tif = "green_denoised_test.tif"


class Parameters():
    def __init__(self):
        self.denoise_crop_size = (352, 576)
        self.denoise_crop_offset = (None, None)
        self.denoise_tmp_data_dir = os.path.join(os.path.expanduser("~"), "tmp", "deepinterpolation", "data")
        self.denoise_tmp_data_name = self.make_denoise_data_name
        self.denoise_tmp_run_dir = os.path.join(os.path.expanduser("~"), "tmp", "deepinterpolation", "runs")
        self.denoise_runid = self.make_runid
        self.denoise_final_dir = "denoising_run"
        self.denoise_delete_tmp_run_dir = True
        self.denoise_params = denoise.DefaultInterpolationParams()
        self.denoise_params.limited_ram = 320*320
        self.denoise_train_each_trial = False
        self.denoise_train_trial = 0

        self.dff_common_baseline = True
        self.dff_baseline_blur = 10
        self.dff_baseline_med_filt = 1
        self.dff_baseline_blur_pre = True
        self.dff_baseline_mode = "convolve"
        self.dff_baseline_length = 10
        self.dff_baseline_quantile = 0.01  # 0.95
        self.dff_use_crop = None   # [128, 608, 80, 400]
        self.dff_manual_add_to_crop = 20
        self.dff_blur = 0
        self.dff_min_baseline = 0
        self.dff_baseline_exclude_trials = None

        self.input_name = "green_com_warped.tif"
        self.output_base_name = "green_denoised_test.tif"
        self.dff_baseline_name = "dff_baseline.tif"
        self.dff_name = "dff.tif"
        self.out_base_dir = "/mnt/NAS2/JB/longterm/deep_interpolation_test"
        self.video_base_name = "video_compare.mp4"
        self.dff_video_base_name = "dff_video.mp4"

        self.param_names = None
        self.param_values = None

    def change_denoise_params(self, param_names, param_values):
        self.param_names = param_names
        self.param_values = param_values
        for param_name, param_value in zip(param_names, param_values):
            if param_name == "N_trials":
                self.N_trials = param_value
            elif param_name == "N_frames_per_trial":
                self.denoise_params.N_frames_per_trial = param_value
            elif param_name == "nb_times_through_data":
                self.denoise_params.nb_times_through_data = param_value

    @staticmethod
    def make_runid(trial_dir, appendix=None):
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = date + "_" + fly + "_" + trial
        return (out + "_" + appendix) if appendix is not None else out

    def make_run_name(self, fly_dir, appendix=None):
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = date + "_" + fly
        if self.param_values is not None:
            for value in self.param_values:
                out += f"_{value}"
        return (out + "_" + appendix) if appendix is not None else out

    def make_output_name(self, trial_dir, appendix=None):
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = self.output_base_name[:-4] + "_" +  date + "_" + fly + "_" + trial
        if self.param_values is not None:
            for value in self.param_values:
                out += f"_{value}"
        if appendix is not None:
            out += appendix
        out += ".tif"
        return out
    
    def make_dff_baseline_name(self, trial_dir, appendix=None, notrial=False):
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = self.dff_baseline_name[:-4] + "_" +  date + "_" + fly 
        if not notrial:
            out += "_" 
            out += trial
        if self.param_values is not None:
            for value in self.param_values:
                out += f"_{value}"
        if appendix is not None:
            out += appendix
        out += ".tif"
        return out

    def make_dff_name(self, trial_dir, appendix=None):
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = self.dff_name[:-4] + "_" +  date + "_" + fly + "_" + trial
        if self.param_values is not None:
            for value in self.param_values:
                out += f"_{value}"
        if appendix is not None:
            out += appendix
        out += ".tif"
        return out

    def make_dff_video_name(self, fly_dir, appendix=None):
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = self.dff_video_base_name[:-4] + "_" +  date + "_" + fly
        if self.param_values is not None:
            for value in self.param_values:
                out += f"_{value}"
        if appendix is not None:
            out += appendix
        out += ".mp4"
        return out

    def make_video_name(self, trial_dir, appendix=None):
        fly_dir, trial = os.path.split(trial_dir)
        date_dir, fly = os.path.split(fly_dir)
        _, date = os.path.split(date_dir)
        out = self.output_base_name[:-4] + "_" +  date + "_" + fly + "_" + trial 
        if appendix is not None:
            out += appendix
        out += ".mp4"
        return out

    def make_video_text(self, appendix=None):
        out = ""
        if self.param_values is not None:
            for value, name in zip(self.param_values, self.param_names):
                if name == "N_trials":
                    out += f"N_tr:{value} "
                elif name == "N_frames_per_trial":
                    out += f"N_fr:{value} "
                elif name == "nb_times_through_data":
                    out += f"N_x:{value} "
        if appendix is not None:
            out += appendix
        return out

    def make_denoise_data_name(self, trial_dir):
        return self.make_runid(trial_dir, appendix=self.input_name)

def get_illumination_correction(fly_dir):
    pkl_dir = os.path.join(fly_dir, "processed", "compare_trials.pkl")
    with open(pkl_dir, "rb") as f:
        summary_dict = pickle.load(f)
    greens = summary_dict["green_means_raw"]
    del summary_dict
    green = np.mean(greens, axis=0)
    green_med_filt = medfilt(green, kernel_size=(71,91))
    green_filt = gaussian_filter(green_med_filt, sigma=3)

    # select the area +-100 pixels from the center
    y_mean = np.mean(green_filt, axis=0)
    norm_range = [len(y_mean) // 2 - 100, len(y_mean) // 2 + 100]

    # perform linear regression in that range
    y_target = y_mean[norm_range[0]:norm_range[1]]
    x_target = np.arange(norm_range[0], norm_range[1])
    # model: y = b[0]x + b[1]
    X = np.hstack((np.expand_dims(x_target, axis=1), np.ones((len(x_target),1))))
    b = np.linalg.pinv(X.T).T.dot(y_target)
    print(f"{fly_dir}\nFound correction parameters: offset={b[1]}, slope={b[0]}")
    correction = 1/(1 +  b[0]/b[1]*np.arange(len(y_mean)))
    return correction

def correct_illumination(stack, correction):
    return stack*correction


def main_runs():
    params = Parameters()
    downsample = 1  # 2
    correct = True
    correct_after = False
    already_trained = True
    alread_prepared = True
    for i_fly, (fly_dir, trials) in enumerate(zip(fly_dirs, all_selected_trials)):
        if i_fly < 2:
            continue
        print(time.ctime(time.time()), fly_dir)
        all_trial_dirs = utils.readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
        trial_dirs = [all_trial_dirs[i] for i in trials]
        # prepare training and test data
        input_datas = [os.path.join(trial_dir, load.PROCESSED_FOLDER, params.input_name)
                                for trial_dir in trial_dirs]
        tmp_data_dirs = [os.path.join(params.denoise_tmp_data_dir,
                            params.make_runid(trial_dir))
                        for trial_dir in trial_dirs]
        tmp_data_downsamp_dirs = [os.path.join(params.denoise_tmp_data_dir,
                            params.make_runid(trial_dir, appendix="downsample"))
                        for trial_dir in trial_dirs]
        tmp_data_correct_dirs= [os.path.join(params.denoise_tmp_data_dir,
                            params.make_runid(trial_dir, appendix="corrected"))
                        for trial_dir in trial_dirs]
        
        if (not correct or correct_after) and not alread_prepared:
            denoise.prepare_data(train_data_tifs=input_datas,
                                out_data_tifs=tmp_data_dirs,
                                offset=params.denoise_crop_offset,
                                size=params.denoise_crop_size)

        def prepare_downsample_data(train_data_tifs, out_data_tifs, downsample=2):
            if not isinstance(train_data_tifs, list):
                train_data_tifs = [train_data_tifs]
            if not isinstance(out_data_tifs, list):
                out_data_tifs = [out_data_tifs]
            assert len(train_data_tifs) == len(out_data_tifs)

            stacks = [utils.get_stack(path) for path in train_data_tifs]
            N_frames, N_y, N_x = stacks[0].shape
            image_size = [N_y, N_x]

            if all(np.array(image_size) % (32*downsample) == 0):
                stacks = [stack[:,::downsample,::downsample] for stack in stacks]
            else:
                mod_y = N_y % (32*downsample)
                N_y_new = N_y - mod_y
                mod_x = N_x % (32*downsample)
                N_x_new = N_x - mod_x
                stacks = [stack[:, mod_y//2:mod_y//2+N_y_new:downsample, mod_x//2:mod_x//2+N_x_new:downsample] 
                          for stack in stacks]
            for out_path, stack in zip(out_data_tifs, stacks):
                path, _ = os.path.split(out_path)
                if not os.path.isdir(path):
                    os.makedirs(path)
                utils.save_stack(out_path, stack)
        if (downsample > 1 and not correct) and not alread_prepared:
            prepare_downsample_data(train_data_tifs=input_datas,
                                    out_data_tifs=tmp_data_downsamp_dirs,
                                    downsample=downsample)

        def prepare_corrected_data(train_data_tifs, out_data_tifs, fly_dir):
            if not isinstance(train_data_tifs, list):
                train_data_tifs = [train_data_tifs]
            if not isinstance(out_data_tifs, list):
                out_data_tifs = [out_data_tifs]
            assert len(train_data_tifs) == len(out_data_tifs)

            stacks = [utils.get_stack(path) for path in train_data_tifs]

            correction = get_illumination_correction(fly_dir)
            stacks_corrected = [correct_illumination(stack, correction) for stack in stacks]
            for out_path, stack in zip(out_data_tifs, stacks_corrected):
                path, _ = os.path.split(out_path)
                if not os.path.isdir(path):
                    os.makedirs(path)
                utils.save_stack(out_path, stack)
        if correct and not alread_prepared:
            prepare_corrected_data(train_data_tifs=input_datas,
                                   out_data_tifs=tmp_data_correct_dirs,
                                   fly_dir=fly_dir)

        param_names = list(parameter_options.keys())
        param_values = list(parameter_options.values())
        param_combos = list(itertools.product(*param_values))
        for i_c, param_combo in enumerate(param_combos):
            # if i_fly == 2 and i_c < 2:
            #     continue
            if i_c not in [2]:  # 0,1,2,4]:
                continue
            print(time.ctime(time.time()), "=====================================")
            for param_name, param_value in zip(param_names, param_combo):
                print(f"{param_name}: {param_value}")
            N_trials = param_combo[0]
            params.change_denoise_params(param_names, param_values=param_combo)
            if downsample > 1 and not correct:
                params.denoise_params.generator_name = "SingleTifGenerator"
                train_data_tifs = [tmp_data_downsamp_dirs[i] for i in denoise_training_trials[:N_trials]]
                test_data_tifs = [tmp_data_downsamp_dirs[i] for i in denoise_test_trials]
                run_identifier = params.make_run_name(fly_dir, appendix=f"_down{downsample}")
            elif not correct or correct_after:
                params.denoise_params.generator_name = "SingleTifGeneratorRandomX"
                train_data_tifs = [tmp_data_dirs[i] for i in denoise_training_trials[:N_trials]]
                test_data_tifs = [tmp_data_dirs[i] for i in denoise_test_trials]
                run_identifier = params.make_run_name(fly_dir)
            elif correct:
                params.denoise_params.generator_name = "SingleTifGeneratorRandomX"
                train_data_tifs = [tmp_data_correct_dirs[i] for i in denoise_training_trials[:N_trials]]
                test_data_tifs = [tmp_data_correct_dirs[i] for i in denoise_test_trials]
                run_identifier = params.make_run_name(fly_dir, appendix="_correct")

            run_base_dir = params.denoise_tmp_run_dir
            
            if not already_trained:
                tmp_run_dir = denoise.train(train_data_tifs=train_data_tifs,
                                            run_base_dir=run_base_dir,
                                            run_identifier=run_identifier,
                                            params=params.denoise_params)
            else:
                run_dirs = glob(os.path.join(params.denoise_tmp_run_dir, run_identifier+"unet*"))
                tmp_run_dir = sorted(run_dirs)[-1]
                print("Using already trained model:", tmp_run_dir)
            if downsample > 1 and not correct:
                tif_out_dirs = [os.path.join(params.out_base_dir, params.make_output_name(trial_dir, appendix=f"_down{downsample}"))
                                for trial_dir in trial_dirs]
            elif not correct and not correct_after:
                tif_out_dirs = [os.path.join(params.out_base_dir, params.make_output_name(trial_dir))
                                for trial_dir in trial_dirs]
            elif correct and not correct_after:
                tif_out_dirs = [os.path.join(params.out_base_dir, params.make_output_name(trial_dir, appendix="_correct"))
                                for trial_dir in trial_dirs]
                tmp_data_dirs = tmp_data_correct_dirs
            elif correct_after:
                tif_out_dirs = [os.path.join(params.out_base_dir, params.make_output_name(trial_dir, appendix="_correct_after"))
                                for trial_dir in trial_dirs]
                tmp_data_dirs = tmp_data_correct_dirs
            denoise.inference(data_tifs=tmp_data_dirs,
                                run_dir=tmp_run_dir,
                                tif_out_dirs=tif_out_dirs,
                                params=params.denoise_params)
            # denoise.clean_up(tmp_run_dir, tmp_data_dirs)

def main_videos():
    params = Parameters()
    i_fly = 2  # 0
    i_trial = 0  # 0
    fly_dir = fly_dirs[i_fly]
    i_trial = all_selected_trials[i_fly][i_trial]
    all_trial_dirs = utils.readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
    trial_dir = all_trial_dirs[i_trial]

    green_raw_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER, params.input_name)
    red_raw_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER, "red_com_warped.tif")
    green_corrected_dir = os.path.join(params.denoise_tmp_data_dir,
                            params.make_runid(trial_dir, appendix="corrected"))
    param_names = list(parameter_options.keys())
    param_values = list(parameter_options.values())
    param_combos = list(itertools.product(*param_values))

    tif_out_dirs = []
    video_texts = ["raw"]
    for i_c, param_combo in enumerate(param_combos):
        if i_c not in [0,1,2,4]:
            continue
        params.change_denoise_params(param_names, param_values=param_combo)

        tif_out_dir = os.path.join(params.out_base_dir, params.make_output_name(trial_dir))
        tif_out_dirs.append(tif_out_dir)
        video_texts.append(params.make_video_text())

    # show the one with downsampled training data
    param_combo = param_combos[2]
    params.change_denoise_params(param_names, param_values=param_combo)
    tif_out_dir = os.path.join(params.out_base_dir,
                               params.make_output_name(trial_dir, appendix=f"_down{2}"))
    tif_out_dirs.append(tif_out_dir)
    video_texts.append(params.make_video_text(appendix=" down:2"))

    # show the raw data with illumination correction
    tif_out_dirs.append(green_corrected_dir)
    video_texts.append("raw corrected")

    # show the denoised one with illumination correction
    tif_out_dir = os.path.join(params.out_base_dir,
                               params.make_output_name(trial_dir, appendix="_correct"))
    tif_out_dirs.append(tif_out_dir)
    video_texts.append(params.make_video_text(appendix=" correct"))

    reds = [red_raw_dir] + [None for _ in tif_out_dirs]
    greens = [green_raw_dir] + tif_out_dirs
    make_2p_grid_video(greens=greens, reds=reds, out_dir=params.out_base_dir,
                       video_name=params.make_video_name(trial_dir, appendix="correct"),
                       percentiles=(5,99),
                       frame_rate=None, trial_dir=trial_dir,
                       texts=video_texts, force_N_frames=16*30)

def main_dff():
    params = Parameters()
    param_names = list(parameter_options.keys())
    param_values = list(parameter_options.values())
    param_combos = list(itertools.product(*param_values))

    dff_param_names = list(dff_parameter_options.keys())
    dff_param_values = list(dff_parameter_options.values())
    dff_param_combos = list(itertools.product(*dff_param_values))

    all_inputs = []
    i_proc = 0
    for i_fly, (fly_dir, selected_trials) in enumerate(zip(fly_dirs, all_selected_trials)):
        for i_dff, dff_param_combo in enumerate(dff_param_combos):
            if i_dff == 0:
                continue
            # if i_dff == 1 and i_fly == 0:
            #     continue
            if not (i_fly == 0 and dff_param_combo[0] == True and dff_param_combo[1] == 20):
                continue

            this_inputs = {
                "i_proc": i_proc,
                "fly_dir": fly_dir,
                "selected_trials": selected_trials,
                "params": params,
                "param_combo": param_combos[2],
                "param_names": param_names,
                "dff_param_combo": dff_param_combo,
            }
            i_proc += 1
            all_inputs.append(this_inputs)

    
    
    pool = multiprocessing.Pool(processes=5)
    pool.map(multiproc_task, all_inputs)

def multiproc_task(inputs):
    i_proc = inputs["i_proc"]
    fly_dir = inputs["fly_dir"]
    selected_trials = inputs["selected_trials"]
    params = inputs["params"]
    param_combo = inputs["param_combo"]
    param_names = inputs["param_names"]
    dff_param_combo = inputs["dff_param_combo"]
    params = deepcopy(params)
    print(i_proc, time.ctime(time.time()), fly_dir)
    all_trial_dirs = utils.readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
    trial_dirs = [all_trial_dirs[i_trial] for i_trial in selected_trials]

    params.change_denoise_params(param_names, param_values=param_combo)
    stacks = [os.path.join(params.out_base_dir,
                params.make_output_name(trial_dir, appendix="_correct"))
                for trial_dir in trial_dirs]

    # for i_dff, dff_param_combo in enumerate(tqdm(dff_param_combos)):
    #     if i_dff == 0:
    #         continue
    print(i_proc, dff_param_combo)
    dff_param_appendix = f"_{dff_param_combo[0]}_{dff_param_combo[1]}"
    if dff_param_combo[0]:
        params.dff_baseline_mode = "quantile"
    else:
        params.dff_baseline_mode = "convolve"
    params.dff_baseline_length = dff_param_combo[1]
    # video_text = params.make_video_text(appendix=" correct")
    baseline_dirs = [os.path.join(params.out_base_dir,
                        params.make_dff_baseline_name(trial_dir, appendix="_correct"+dff_param_appendix))
                        for trial_dir in trial_dirs]
    common_baseline_dir = os.path.join(params.out_base_dir,
                                        params.make_dff_baseline_name(trial_dirs[0], 
                                        appendix="_correct"+dff_param_appendix,
                                        notrial=True))


    print(i_proc, time.ctime(time.time()), "computing dff baseline")
    if not os.path.isfile(common_baseline_dir):
        _ = dff.find_dff_baseline_multi_stack_load_single(
            stacks=stacks,
            individual_baseline_dirs=baseline_dirs,
            baseline_blur=params.dff_baseline_blur,
            baseline_med_filt=params.dff_baseline_med_filt,
            blur_pre=params.dff_baseline_blur_pre,
            baseline_mode=params.dff_baseline_mode,
            baseline_length=params.dff_baseline_length,
            baseline_quantile=params.dff_baseline_quantile,
            min_baseline=params.dff_min_baseline,
            baseline_dir=common_baseline_dir
            )

    for stack, trial_dir in zip(stacks, trial_dirs):
        print(i_proc, time.ctime(time.time()), "computing dff for stack: " + stack)
        if not os.path.isfile(os.path.join(params.out_base_dir, 
                                        params.make_dff_name(trial_dir, appendix="_correct"+dff_param_appendix))):
            _ = dff.compute_dff_from_stack(
                stack,
                baseline_blur=0,
                baseline_med_filt=0,
                blur_pre=False,
                baseline_mode="fromfile",
                baseline_length=params.dff_baseline_length,
                baseline_quantile=params.dff_baseline_quantile,
                baseline_dir=common_baseline_dir,
                use_crop=params.dff_use_crop,
                manual_add_to_crop=params.dff_manual_add_to_crop,
                dff_blur=params.dff_blur,
                min_baseline=params.dff_min_baseline,
                dff_out_dir=os.path.join(params.out_base_dir, 
                                            params.make_dff_name(trial_dir, appendix="_correct"+dff_param_appendix)),
                return_stack=False
                )
    

def main_dff_video():
    params = Parameters()
    param_names = list(parameter_options.keys())
    param_values = list(parameter_options.values())
    param_combos = list(itertools.product(*param_values))
    param_combo = param_combos[2]
    params.change_denoise_params(param_names, param_values=param_combo)

    dff_param_names = list(dff_parameter_options.keys())
    dff_param_values = list(dff_parameter_options.values())
    dff_param_combos = list(itertools.product(*dff_param_values))

    for i_fly, (fly_dir, selected_trials) in enumerate(zip(fly_dirs, all_selected_trials)):
        if not i_fly == 0:
            continue
        print(time.ctime(time.time()), fly_dir)
        all_trial_dirs = utils.readlines_tolist(os.path.join(fly_dir, "trial_dirs.txt"))
        trial_dirs = [all_trial_dirs[i_trial] for i_trial in selected_trials]
        dff_orig_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER, "dff_denoised_t1.tif")
                            for trial_dir in trial_dirs]
        dff_correct_dirs = [os.path.join(params.out_base_dir,
                                        params.make_dff_name(trial_dir, appendix="_correct"))
                            for trial_dir in trial_dirs]


        dff_param_dirs = [[os.path.join(params.out_base_dir,
                                        params.make_dff_name(trial_dir, appendix="_correct" + f"_{dff_param_combo[0]}_{dff_param_combo[1]}"))
                            for trial_dir in trial_dirs]
                            for dff_param_combo in dff_param_combos[1:]]

        orig_names = [trial_dir.split("/")[-1] + " old" for trial_dir in trial_dirs]
        correct_names = [trial_dir.split("/")[-1] + " corrected" for trial_dir in trial_dirs]

        dff_names = [[trial_dir.split("/")[-1] + " corrected" + f" {dff_param_combo[0]} {dff_param_combo[1]}"
                      for trial_dir in trial_dirs]
                      for dff_param_combo in dff_param_combos[1:]]

        make_multiple_video_dff(
            dffs=[dff_orig_dirs, dff_correct_dirs] + dff_param_dirs,
            out_dir=params.out_base_dir,
            video_name=params.make_dff_video_name(fly_dir, appendix="_dff_param"),
            frame_rate=None, trial_dir=trial_dirs[0],
            vmin=0, vmax=None, pmin=1, pmax=99, share_lim=False,
            blur=0, crop=None, mask=None, share_mask=False,
            text=[orig_names, correct_names] + dff_names,
            frames=np.arange(130*16,16*160))
if __name__ == "__main__":
    # main_runs()
    # main_videos()
    # main_dff()
    main_dff_video()
