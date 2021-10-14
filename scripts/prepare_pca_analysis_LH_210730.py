import os, sys
from copy import deepcopy

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp import load, utils
from twoppp.pipeline import PreProcessFly, PreProcessParams

if __name__ == "__main__":

    params = PreProcessParams()
    params.genotype = "J1xM5"
    
    params.breadth_first = True
    params.overwrite = False

    params.green_denoised = "green_denoised_t1.tif"
    

    fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # high caff
                os.path.join(load.NAS2_DIR_LH, "210721", "fly3"),  # high caff
                os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
                os.path.join(load.NAS2_DIR_LH, "210723", "fly2")   # high caff
                ]
    all_selected_trials = [[1,4,5,8,10,12],
                           [1,4,5,8,10,12],
                           [1,4,5,8,10,12],
                           [1,5,6,9,11,12]]

    conditions = ["210722 fly 3 high caff",
                  "210721 fly 3 high caff",
                  "210723 fly 1 low caff",
                  "210723 fly 2 high caff"]
  
    params_copy = deepcopy(params)

    for i_fly, (fly_dir, selected_trials, condition) in \
        enumerate(zip(fly_dirs, all_selected_trials, conditions)):
        params = deepcopy(params_copy)   

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params, trial_dirs="fromfile", 
                                   selected_trials=selected_trials,
                                   beh_trial_dirs="fromfile", sync_trial_dirs="fromfile")
        
        preprocess.get_dfs()

        # preprocess.extract_rois()

        preprocess.prepare_pca_analysis(condition=condition, load_df=False, load_pixels=True)



            
            

       
    
