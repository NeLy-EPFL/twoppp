import os.path
from . import load

fly_dirs = [
    os.path.join(load.NAS2_DIR_LH, "210722", "fly3"),  # high caff
    os.path.join(load.NAS2_DIR_LH, "210721", "fly3"),  # high caff
    os.path.join(load.NAS2_DIR_LH, "210723", "fly1"),  # low caff
    os.path.join(load.NAS2_DIR_LH, "210723", "fly2"),  # high caff
    os.path.join(load.NAS2_DIR_LH, "210718", "fly2"),  # starv

    os.path.join(load.NAS2_DIR_LH, "210719", "fly1"),  # sucr
    os.path.join(load.NAS2_DIR_LH, "210719", "fly2"),  # starv
    os.path.join(load.NAS2_DIR_LH, "210719", "fly4"),  # sucr
    os.path.join(load.NAS2_DIR_LH, "210802", "fly1"),  # lowcaff
    os.path.join(load.NAS2_DIR_LH, "210804", "fly1"),  # low caff

    os.path.join(load.NAS2_DIR_LH, "210804", "fly2"),  # low caff
    os.path.join(load.NAS2_DIR_LH, "210811", "fly2"),  # high caff
    os.path.join(load.NAS2_DIR_LH, "210811", "fly1"),  # starv
    os.path.join(load.NAS2_DIR_LH, "210812", "fly1"),  # starv
    os.path.join(load.NAS2_DIR_LH, "210813", "fly1"),  # sucr

    os.path.join(load.NAS2_DIR_LH, "210818", "fly3"),  # sucr
    os.path.join(load.NAS2_DIR_LH, "210901", "fly1"),  # starv
    os.path.join(load.NAS2_DIR_LH, "210902", "fly2"),  # sucr
]

all_selected_trials = [
    [1,3,4,5,6,7,8,9,10,11,12,13],
    [1,3,4,5,6,7,8,9,10,11,12],
    [1,3,4,5,6,8,9,10,11,12],
    [1,3,5,6,7,8,9,10,11,12,14],  # 13 excluded because no camera-metadata
    [2,4,5,7],

    [2,4,5,7,8],
    [2,4,5,7],
    [2,4,5,7],
    [1,3,4,5,6,7,8,9,11,12],  # 10 exlcuded because CC out of center
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