import os.path
from . import load
from . import utils

BASE_DIR = load.NAS2_DIR_LH

fly_dirs = [
    os.path.join(BASE_DIR, "210722", "fly3"),  # high caff
    os.path.join(BASE_DIR, "210721", "fly3"),  # high caff
    os.path.join(BASE_DIR, "210723", "fly1"),  # low caff
    os.path.join(BASE_DIR, "210723", "fly2"),  # high caff
    os.path.join(BASE_DIR, "210718", "fly2"),  # starv

    os.path.join(BASE_DIR, "210719", "fly1"),  # sucr
    os.path.join(BASE_DIR, "210719", "fly2"),  # starv
    os.path.join(BASE_DIR, "210719", "fly4"),  # sucr
    os.path.join(BASE_DIR, "210802", "fly1"),  # lowcaff
    os.path.join(BASE_DIR, "210804", "fly1"),  # low caff

    os.path.join(BASE_DIR, "210804", "fly2"),  # low caff
    os.path.join(BASE_DIR, "210811", "fly2"),  # high caff
    os.path.join(BASE_DIR, "210811", "fly1"),  # starv
    os.path.join(BASE_DIR, "210812", "fly1"),  # starv
    os.path.join(BASE_DIR, "210813", "fly1"),  # sucr

    os.path.join(BASE_DIR, "210818", "fly3"),  # sucr
    os.path.join(BASE_DIR, "210901", "fly1"),  # starv
    os.path.join(BASE_DIR, "210902", "fly2"),  # sucr
]

all_selected_trials = [
    [1,3,4,5,6,7,8,9,10,11,12,13],
    [1,3,4,5,6,7,8,9,10,11,12],
    [1,3,4,5,6,8,9,10,11,12],
    [1,3,5,6,7,8,9,10,11,12,14],  # 13 excluded because no camera-metadata
    [2,3,4,5,6,7,8,9,10],  # finishes at cs_starv_after_005

    [2,3,4,5,6,7,8,9,10],  # finishes at cs_sucr_after_005
    [2,3,4,5,6,7,8,9,10],  # finishes at cs_sucr_after_005
    [2,4,5,7],  # don't use
    [1,3,4,5,6,7,8,9,9,11,12],  # 10 excluded because CC out of center
    [1,3,4,5,6,7,8,9,10,11,12], # don't use

    [1,3,5,6,7,8,9,10,11,12],  # 4 excluded for now, because processing not yet ready
    [0,2,5,6,7,8,9,10,11,12,13,16], # don't use
    [2,4,5,7], # don't use
    [2,3,4,5,6,7,8,9,10,11,12],
    [2,3,5,6,7,8,9,10,11,12,13],

    [2,3,4,5,6,7,8,9,10,11,12],
    [2,3,4,5,6,7,8,9,10,11,12],
    [2,4,5,7],  # don't use
]

all_selected_trials_old = [
    [1,3,4,5,6,7,8,9,10,11,12,13],
    [1,3,4,5,6,7,8,9,10,11,12],
    [1,3,4,5,6,8,9,10,11,12],
    [1,3,5,6,7,8,9,10,11,12,14],  # 13 excluded because no camera-metadata
    [2,4,5,7],

    [2,4,5,7,8],
    [2,4,5,7],
    [2,4,5,7],
    [1,3,4,5,6,7,8,9,11,12],  # 10 excluded because CC out of center
    [1,3,4,5,6,7,8,9,10,11,12],

    [1,3,5,6,7,8,9,10,11,12],  # 4 excluded for now, because processing not yet ready
    [0,2,5,6,7,8,9,10,11,12,13,16],
    [2,4,5,7],
    [2,4,5,7],
    [2,5,6,8],

    [2,4,5,7],
    [2,4,5,7],
    [2,4,5,7],
]

conditions = [
    "210722 fly 3 high caff",
    "210721 fly 3 high caff",
    "210723 fly 1 low caff",
    "210723 fly 2 high caff", #TODO: trial one red_com_warped is weirdly smaller than the others
    "210718 fly 2 starv",

    "210719 fly 1 sucr",
    "210719 fly 2 starv",
    "210719 fly 4 sucr",
    "210802 fly 1 low caff",
    "210804 fly 1 low caff",

    "210804 fly 2 low caff",
    "210811 fly 2 high caff",
    "210811 fly 1 starv",
    "210812 fly 1 starv",
    "210813 fly 1 sucr",

    "210818 fly 3 sucr",
    "210901 fly 1 starv",
    "210902 fly 2 sucr",
]

class Trial():
    def __init__(self, fly, trial_dir, beh_trial_dir, sync_trial_dir):
        super().__init__()
        self.fly = fly
        self.dir = trial_dir
        self.beh_dir = beh_trial_dir
        self.sync_dir = sync_trial_dir
        self.processed_dir = os.path.join(self.dir, load.PROCESSED_FOLDER)
        self.green_denoised = os.path.join(self.processed_dir, "green_denoised_t1_corr.tif")
        self.green_raw = os.path.join(self.processed_dir, "green_com_warped.tif")
        self.red_raw = os.path.join(self.processed_dir, "red_com_warped.tif")
        self.green_norm = os.path.join(self.processed_dir, "green_denoised_t1_corr_norm.tif")
        self.dff = os.path.join(self.processed_dir, "dff_denoised_t1_corr.tif")
        self.name = self.dir.split("/")[-1].replace("_", " ")
        self.twop_df = os.path.join(self.processed_dir, "twop_df.pkl")


class Fly():
    def __init__(self, i_fly):
        super().__init__()
        self.i_fly = i_fly
        self.dir = fly_dirs[self.i_fly]
        self.i_flyonday = int(self.dir.split("/")[-1][-1:])
        self.date = int(self.dir.split("/")[-2])
        self.processed_dir = os.path.join(self.dir, load.PROCESSED_FOLDER)
        self.selected_trials = all_selected_trials[self.i_fly]
        self.condition = conditions[self.i_fly]
        self.paper_condition = self.get_paper_condition(self.condition)
        
        trial_dirs = utils.readlines_tolist(os.path.join(self.dir, "trial_dirs.txt"))
        beh_dirs = utils.readlines_tolist(os.path.join(self.dir, "beh_trial_dirs.txt"))
        sync_dirs = utils.readlines_tolist(os.path.join(self.dir, "sync_trial_dirs.txt"))
        self.trials = [Trial(self,
                             trial_dirs[i_trial],
                             beh_dirs[i_trial],
                             sync_dirs[i_trial])
                       for i_trial in self.selected_trials]
        self.mask_coarse = os.path.join(self.processed_dir, "cc_mask.tif")
        self.fs = 16
        self.summary_dict = os.path.join(self.processed_dir, "compare_trials.pkl")
        wave_mask_names = ["mask_top.tif", "mask_left.tif", "mask_right.tif",
                           "mask_bottom.tif", "mask_gf_left.tif", "mask_gf_right.tif"]
        self.wave_masks = [os.path.join(self.processed_dir, wave_mask) for wave_mask in wave_mask_names]
        self.trials_mean_dff = os.path.join(self.processed_dir, "trials_mean_dff.pkl")  # trials_mean_dff_raw
        self.raw_std = os.path.join(self.processed_dir, "raw_std.tif")
        self.mask_fine = os.path.join(self.processed_dir, "cc_mask_fiji.tif")
        self.wave_details = os.path.join(self.processed_dir, "wave_details.pkl")  # wave_details_raw
        self.rest_maps = os.path.join(self.processed_dir, "rest_maps.pkl")
        self.correction = os.path.join(self.processed_dir, "illumination_correction.pkl")

    @staticmethod
    def get_paper_condition(condition):
        if "high caff" in condition:
            return "high caffeine"
        elif "low caff" in condition:
            return "low caffeine"
        elif "sucr" in condition:
            return "sucrose"
        elif "starv" in condition:
            return "starved"
        else:
            return ""
    @property
    def trial_dirs(self):
        return [trial.dir for trial in self.trials]

    @property
    def trial_processed_dirs(self):
        return [trial.processed_dir for trial in self.trials]

    @property
    def trial_names(self):
        return [trial.name for trial in self.trials]


high_caff_flies = [Fly(i_fly) for i_fly in [0, 1, 3]]
high_caff_main_fly = high_caff_flies[0]

low_caff_flies = [Fly(i_fly) for i_fly in [8, 10, 2]]
low_caff_main_fly = low_caff_flies[0]

sucr_flies = [Fly(i_fly) for i_fly in [15, 5]]
sucr_main_fly = sucr_flies[0]

starv_flies = [Fly(i_fly) for i_fly in [16, 4, 6]]
starv_main_fly = starv_flies[0]

all_flies = high_caff_flies + low_caff_flies + sucr_flies + starv_flies
main_flies = [high_caff_main_fly, low_caff_main_fly, sucr_main_fly, starv_main_fly]
