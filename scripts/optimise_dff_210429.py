import os, sys
import numpy as np

from copy import copy
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
import pickle
import time
from tqdm import tqdm

from deepinterpolation import interface as denoise
import utils_video
import utils2p

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from longterm import load, utils
import longterm
from longterm.plot import videos
from longterm import dff as _dff

FA_R57C10_trial_dir = os.path.join(load.LOCAL_DATA_DIR, "181220_Rpr_R57C10_GC6s_tdTom", "Fly5", "002_coronal")
FA_ABO_trial_dir = os.path.join(load.LOCAL_DATA_DIR, "201014_G23xU1", "Fly1", "005_coronal")
JB_R57C10_trial_dir = os.path.join(load.LOCAL_DATA_DIR, "210301_J1xCI9", "Fly1", "002_xz")
LH_R57C10_trial_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212", "Fly1", "cs_003")
trial_dirs = [FA_R57C10_trial_dir, FA_ABO_trial_dir, JB_R57C10_trial_dir, LH_R57C10_trial_dir]
trial_names = ["FA_R57C10", "FA_ABO", "JB_R57C10", "LH_R57C10"]

if __name__ == "__main__":
    pre_baseline_med = [(1,1,1), (5,1,1), (5,5,5), (9,9,9)]
    pre_baseline_gauss = [(0,0,0), (0,1,1), (0,3,3), (0,10,10)]
    baseline_length = [10, 20]
    post_baseline_med = [(1,1), (5,5)]
    post_baseline_gauss = [(0,0), (1,1), (3,3)]

    out_dir = os.path.join(load.LOCAL_DATA_DIR, "outputs", "210428_dff")
    step0 = utils.get_stack(os.path.join(trial_dirs[0], "processed_FA","green_warped.tif"))
    step0 = step0[:, 80:-80, 40:-40]

    reference = False
    if reference:
        dff_FA_R57C10_dir = os.path.join(trial_dirs[0], "processed_FA", "dff.tif")
        dff_FA_R57C10 = utils.get_stack(dff_FA_R57C10_dir)
        dff_FA_R57C10 = dff_FA_R57C10[:, 80:-80, 40:-40]
        videos.make_multiple_video_dff(dffs=dff_FA_R57C10, out_dir=out_dir, video_name="video_default",
                                text=None, frames=np.arange(15*8), frame_rate=8.1,
                                vmin=0, pmax=99, blur=0, share_lim=True)

    compute = False
    if compute: 
        for i_pre_med, pre_med in enumerate(pre_baseline_med):
            print(time.ctime(time.time()), "pre med: ", pre_med)
            step1 = medfilt(step0, kernel_size=pre_med) if i_pre_med else step0
            np.save(os.path.join(out_dir, "step_1_{}.npy".format(i_pre_med)), step1, allow_pickle=True)
            for i_pre_gauss, pre_gauss in enumerate(pre_baseline_gauss):
                print(time.ctime(time.time()), "pre gauss: ", pre_gauss)
                step2 = gaussian_filter(step1, sigma=pre_gauss) if i_pre_gauss else step1
                np.save(os.path.join(out_dir, "step_2_{}_{}.npy".format(i_pre_med, i_pre_gauss)), step2, allow_pickle=True)
                for i_bl, bl in enumerate(baseline_length):
                    print(time.ctime(time.time()), "N_baseline: ", bl)
                    step3 = _dff._find_pixel_wise_baseline(step2, n=bl)
                    np.save(os.path.join(out_dir, "step_3_{}_{}_{}.npy".format(i_pre_med, i_pre_gauss, i_bl)), step3, allow_pickle=True)
                    for i_post_med, post_med in enumerate(post_baseline_med):
                        print(time.ctime(time.time()), "post med: ", post_med)
                        step4 = medfilt(step3, kernel_size=post_med) if i_post_med else step3
                        for i_post_gauss, post_gauss in enumerate(post_baseline_gauss):
                            print(time.ctime(time.time()), "post gauss: ", post_gauss)
                            step5 = gaussian_filter(step4, sigma=post_gauss) if i_post_gauss else step4
                            print(time.ctime(time.time()), "compute dff")
                            step6 = _dff._compute_dff(step0, baseline=step5, apply_filter=True)
                            np.save(os.path.join(out_dir, "step_6_{}_{}_{}_{}_{}.npy".format(i_pre_med, i_pre_gauss, i_bl,
                                                                                            i_post_med, i_post_gauss)), 
                                    step6, allow_pickle=True)
                            print(time.ctime(time.time()), "make video")
                            videos.make_multiple_video_dff(dffs=step6, out_dir=out_dir, video_name="video_{}_{}_{}_{}_{}".format(i_pre_med, i_pre_gauss, i_bl,
                                                                                                                        i_post_med, i_post_gauss),
                                                    text=None, frames=np.arange(15*8), frame_rate=8.1,
                                                    vmin=0, pmax=99, blur=0, share_lim=True)
    
    compare = False
    if compare:
        step0 = step0[:125, :, :]
        to_compare = [[0],      # med filt pre
                      [3],      # gauss pre
                      [0,1],    # baseline length
                      [0,1],    # med filt post
                      [0,1,2]]  # gauss filt post
        in_row = [4]
        in_col = [3,2]


        for i_pre_med, pre_med in enumerate(pre_baseline_med):
            to_compare[0][0] = i_pre_med
            print("=====pre_med: ", pre_med)
            for i_pre_gauss, pre_gauss in enumerate(pre_baseline_gauss):
                to_compare[1][0] = i_pre_gauss
                print("=====pre_gauss: ", pre_gauss)
                dff_grid = []
                dff_names_grid = []
                this_compare = [comp[0] for comp in to_compare]
                for col_value1 in to_compare[2]:
                    this_compare[2] = col_value1
                    step3 = np.load(os.path.join(out_dir, "step_3_{}_{}_{}.npy".format(this_compare[0], 
                                                                                    this_compare[1], 
                                                                                    this_compare[2])))
                    for col_value2 in to_compare[3]:
                        this_compare[3] = col_value2
                        print(time.ctime(time.time()), "post med: ", post_baseline_med[this_compare[3]])
                        step4 = medfilt(step3, kernel_size=post_baseline_med[this_compare[3]])
                        dff_this_row = []
                        names_this_row = []
                        for i_row, row in enumerate(in_row):
                            for row_value in to_compare[row]:
                                this_compare[row] = row_value
                                print(time.ctime(time.time()), "post gauss: ", post_baseline_gauss[this_compare[4]])
                                step5 = gaussian_filter(step4, sigma=post_baseline_gauss[this_compare[4]])
                                print(time.ctime(time.time()), "compute dff")
                                step6 = _dff._compute_dff(step0, baseline=step5, apply_filter=True)
                                
                                dff_file_name = os.path.join(out_dir, "step_6_{}_{}_{}_{}_{}_short.npy".format(this_compare[0],
                                                                                                        this_compare[1],
                                                                                                        this_compare[2],
                                                                                                        this_compare[3],
                                                                                                        this_compare[4]))
                                np.save(os.path.join(out_dir, dff_file_name), step6, allow_pickle=True)
                                dff_this_row.append(step6)
                                dff_name =  "M pre: " + str(pre_baseline_med[this_compare[0]]) + \
                                            "\nG pre: " + str(pre_baseline_gauss[this_compare[1]]) + \
                                            "\nN bl: " + str(baseline_length[this_compare[2]]) + \
                                            "\nM post: " + str(post_baseline_med[this_compare[3]]) + \
                                            "\nG post: " + str(post_baseline_gauss[this_compare[4]])
                                names_this_row.append(dff_name)
                                this_compare[row] = to_compare[row][0]
                        dff_grid.append(dff_this_row)
                        dff_names_grid.append(names_this_row)

                videos.make_multiple_video_dff(dffs=dff_grid, out_dir=out_dir, 
                                        video_name="video_compare_{}_{}".format(to_compare[0][0], to_compare[1][0]),
                                        text=dff_names_grid, frames=np.arange(15*8), frame_rate=8.1,
                                        vmin=0, pmax=99, blur=0, share_lim=False)
        
    denoising = False
    if denoising:
        from longterm.pipeline import PreProcessParams
        params = PreProcessParams()
        params.denoise_crop_size = (320, 640)
        params.denoise_crop_offset = (None, None)
        params.denoise_final_dir = "denoising_run"
        params.denoise_delete_tmp_run_dir = True
        params.denoise_params = denoise.DefaultInterpolationParams()
        # input_data = os.path.join(trial_dirs[0], "processed_FA","green_warped.tif")
        input_data = os.path.join(trial_dirs[3], "processed","green_com_warped.tif")
        tmp_data_dir = os.path.join(params.denoise_tmp_data_dir, 
                                    params.denoise_tmp_data_name(os.path.join(trial_dirs[3], "processed")))
        denoise.prepare_data(train_data_tifs=input_data, 
                            out_data_tifs=tmp_data_dir,
                            offset=params.denoise_crop_offset,
                            size=params.denoise_crop_size)
        tmp_run_dir = denoise.train(train_data_tifs=tmp_data_dir, 
                                    run_base_dir=params.denoise_tmp_run_dir,
                                    run_identifier=params.denoise_runid(os.path.join(trial_dirs[3], "processed")),
                                    params=params.denoise_params)
        denoise.inference(data_tifs=tmp_data_dir, 
                        run_dir=tmp_run_dir,
                        tif_out_dirs=os.path.join(os.path.join(trial_dirs[3], "processed"), params.green_denoised),
                        params=params.denoise_params)
        denoise.clean_up(tmp_run_dir, tmp_data_dir)
        # denoise.copy_run_dir(tmp_run_dir,
        #                         os.path.join(os.path.join(trial_dirs[0], "processed_FA"), params.denoise_final_dir),
        #                         delete_tmp=params.denoise_delete_tmp_run_dir)
    
    compare_datasets = True
    if compare_datasets:
        prepare = True
        if prepare:
            for i_ds, (trial_dir, trial_name) in enumerate(zip(trial_dirs, trial_names)):
                print(time.ctime(time.time()), trial_name)
                if "FA" in trial_name:
                    processed_dir = os.path.join(trial_dir, "processed_FA")
                    processed_out_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
                    green_tif = os.path.join(processed_dir, "green_warped.tif")
                    
                elif "JB" in trial_name or "LH" in trial_name:
                    processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
                    processed_out_dir = processed_dir
                    green_tif = os.path.join(processed_dir, "green_com_warped.tif")
                green_denoised_tif = os.path.join(processed_dir, "green_denoised.tif")
                dff_raw_out = os.path.join(processed_out_dir, "dff.tif")
                dff_denoised_out = os.path.join(processed_out_dir, "dff_denoised.tif")
                dff_raw_baseline_out = os.path.join(processed_out_dir, "dff_baseline.tif")
                dff_denoised_baseline_out = os.path.join(processed_out_dir, "dff_denoised_baseline.tif")
                green = utils.get_stack(green_tif)
                green = green[:, 80:-80, 48:-48]
                green_denoised = utils.get_stack(green_denoised_tif)
                
                print(time.ctime(time.time()), "compute raw dff")
                # green_pre_filt = gaussian_filter(green, sigma=(0, 10,10))
                # baseline = _dff._find_pixel_wise_baseline(green_pre_filt, n=10)
                baseline = utils2p.load_img(dff_raw_baseline_out)
                baseline[baseline<10] = 0
                # utils2p.save_img(dff_raw_baseline_out, baseline)
                dff = _dff._compute_dff(green, baseline, apply_filter=True)
                utils2p.save_img(dff_raw_out, dff)
                
                print(time.ctime(time.time()), "compute denoised dff")
                # green_pre_filt = gaussian_filter(green_denoised, sigma=(0, 10,10))
                # baseline_denoised = _dff._find_pixel_wise_baseline(green_pre_filt, n=10)
                baseline_denoised = utils2p.load_img(dff_denoised_baseline_out)
                baseline_denoised[baseline_denoised<10] = 0
                # utils2p.save_img(dff_denoised_baseline_out, baseline_denoised)
                dff_denoised = _dff._compute_dff(green_denoised, baseline_denoised, apply_filter=True)
                utils2p.save_img(dff_denoised_out, dff_denoised)

        make_video = True
        if make_video:
            dffs = []
            dff_names = []
            masks = []
            for i_ds, (trial_dir, trial_name) in enumerate(zip(trial_dirs, trial_names)):
                if "FA" in trial_name:
                    processed_dir = os.path.join(trial_dir, "processed_FA")
                    processed_out_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
                    green_tif = os.path.join(processed_dir, "green_warped.tif")
                    
                elif "JB" in trial_name or "LH" in trial_name:
                    processed_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER)
                    processed_out_dir = processed_dir
                    green_tif = os.path.join(processed_dir, "green_com_warped.tif")
                green_denoised_tif = os.path.join(processed_dir, "green_denoised.tif")
                dff_raw_out = os.path.join(processed_out_dir, "dff.tif")
                dff_denoised_out = os.path.join(processed_out_dir, "dff_denoised.tif")
                dff_raw_baseline_out = os.path.join(processed_out_dir, "dff_baseline.tif")
                dff_denoised_baseline_out = os.path.join(processed_out_dir, "dff_denoised_baseline.tif")

                this_row = [utils.get_stack(dff_raw_out), utils.get_stack(dff_denoised_out)]
                names_row = [trial_name + " raw", trial_name + " denoised"]
                # raw_mask = _dff.find_dff_mask(utils.get_stack(dff_raw_baseline_out))
                # denoised_mask = _dff.find_dff_mask(utils.get_stack(dff_denoised_baseline_out))
                mask_row = [None, None]  # raw_mask, denoised_mask]
                dffs.append(this_row)
                dff_names.append(names_row)
                masks.append(mask_row)

            videos.make_multiple_video_dff(dffs=dffs, out_dir=out_dir, mask=masks,
                                        video_name="video_compare_datasets_4",
                                        text=dff_names, frames=np.arange(15*8), frame_rate=8.1,
                                        vmin=0, pmax=99, blur=False, share_lim=False)


    optimise_denoised = False
    if optimise_denoised:
        green = utils.get_stack(os.path.join(trial_dirs[-1], "processed", "green_denoised.tif"))
        dffs = []
        dff_names = []
        filts = [1,3,5,10]
        for i_filt, filt in enumerate(tqdm(filts)):
            green_pre_filt = gaussian_filter(green, sigma=(0, filt,filt))
            baseline = _dff._find_pixel_wise_baseline(green_pre_filt, n=10)
            utils2p.save_img(os.path.join(trial_dirs[-1], "processed", "dff_denoised_baseline_prefilt_{}.tif".format(filt)), baseline)
            dff = _dff._compute_dff(green, baseline, apply_filter=True)
            utils2p.save_img(os.path.join(trial_dirs[-1], "processed", "dff_denoised_prefilt_{}.tif".format(filt)), dff)
            dffs.append([dff[:150, :, :]])
            dff_names.append([trial_names[-1]+" denoised G pre: {}".format(filt)])
        
        videos.make_multiple_video_dff(dffs=dffs, out_dir=out_dir, 
                                        video_name="video_compare_denoised",
                                        text=dff_names, frames=np.arange(15*8), frame_rate=8.1,
                                        vmin=0, pmax=99, blur=0, share_lim=False)





                    


        

