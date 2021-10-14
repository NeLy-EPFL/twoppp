import os, sys
import numpy as np
from skimage.morphology import binary_opening, binary_closing, binary_opening

FILE_PATH = os.path.realpath(__file__)
SCRIPT_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

from twoppp.analysis import InterPCAAnalysis, InterPCAAnalysisFromFile
from twoppp import load

fly_dirs = [os.path.join(load.NAS2_DIR_LH, "210512", "fly3"),
            os.path.join(load.NAS2_DIR_LH, "210519", "fly1"),
            os.path.join(load.NAS2_DIR_LH, "210521", "fly1"),
            os.path.join(load.NAS2_DIR_LH, "210526", "fly2"),
            os.path.join(load.NAS2_DIR_LH, "210527", "fly4"),

            os.path.join(load.NAS2_DIR_LH, "210602", "fly2"),
            os.path.join(load.NAS2_DIR_LH, "210603", "fly2")]
conditions = ["12.5. water", "19.5. caff", "21.5. nofeed", "26.5. caff", "27.5. nofeed", " 2.6. sucr", "3.6. sucr"]
all_i_trials = [[0,2,3,5,8,11],  
                [3,8,12],
                [3,7,12],
                [2,7,11],
                [3,6,10],
                
                [3,6,10],
                [1,6,10]]
all_compare_i_trials = [[2,5,11], 
                [3,8,12],
                [3,7,12],
                [2,7,11],
                [3,6,10],
                
                [3,6,10],
                [1,6,10]]
all_thres_rest = [0.01, 0.015, 0.01, 0.01, 0.015, \
                  0.015, 0.015]
all_thres_walk = [0.035, 0.03, 0.025, 0.02, 0.03, \
                  0.04, 0.04]
shapes = [[25, 305, 110, 520],
          [40, 275, 75, 515],
          [40, 290, 90, 510],
          [45, 300, 100, 490],
          [55, 300, 75, 515],
          
          [45, 305, 80, 500],
          [40, 310, 65, 515]]
all_trial_names = [["long before", "medium before", "right before", "right after", "medium after", "long after"],
                   ["medium before", "right after", "long after"],
                   ["medium before", "right after", "long after"],
                   ["medium before", "right after", "long after"],
                   ["medium before", "right after", "long after"],
                   
                   ["medium before", "right after", "long after"],
                   ["medium before", "right after", "long after"]]   

for i_fly, fly_dir in enumerate(fly_dirs):
    if i_fly < 6:
        continue
    print(fly_dir)
    pcan = InterPCAAnalysis(fly_dir=fly_dir, 
                        i_trials=all_i_trials[i_fly], 
                        condition=conditions[i_fly], 
                        compare_i_trials=all_compare_i_trials[i_fly], 
                        thres_walk=all_thres_walk[i_fly], 
                        thres_rest=all_thres_rest[i_fly],
                        load_df=True, 
                        load_pixels=True, 
                        pixel_shape=shapes[i_fly], 
                        sigma=1, # 0  # 1  # 3 
                        trial_names=all_trial_names[i_fly])
    out_file = os.path.join(fly_dir, "processed", "pcan_sigma1.pkl")
    print("pickling")
    pcan.pickle_self(out_file)

if 0:
    pcan = InterPCAAnalysisFromFile(os.path.join(load.NAS2_DIR_LH, "210512", "fly3", "processed", "pcan.pkl"))
    pcan.sort_neurons()
    w = pcan.get_w_inter_pca_walk_rest_neurons()
    pcan.get_covs_walk_rest_neurons()
    pcan.get_covs_inter_trial_neurons()
    pcan.get_inter_pca_trials_neurons()

    w = pcan.get_w_inter_pca_walk_rest_pixels(i_trials=[1]).reshape(pcan.size_y, -1)
    abs_im = (np.abs(w) > np.quantile(np.abs(w),0.95)).astype(np.int32)
    abs_im = binary_opening(abs_im, selem=np.ones((3,2)))
    abs_im = binary_closing(abs_im, selem=np.ones((3,2)))
    selected_pixels = abs_im.flatten()
    pcan.select_pixels(selected_pixels)
    pcan.sort_pixels()
    w = pcan.pixels_to_image(pcan.get_w_inter_pca_walk_rest_pixels())
    pcan.get_covs_walk_rest_pixels()
    pcan.get_covs_inter_trial_pixels()
    pcan.get_inter_pca_trials_pixels()