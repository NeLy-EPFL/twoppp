# Jonas Braun
# jonas.braun@epfl.ch
# 18.02.2021

import os
from pathlib import Path
from shutil import copyfile

import utils2p

NAS_DIR = "/mnt/NAS"
NAS2_DIR = "/mnt/NAS2"
LABSERVER_DIR = "/mnt/labserver"
LABSERVER_DIR_LH = os.path.join(LABSERVER_DIR, "HERMANS_Laura", "Experimental_data")
LABSERVER_DIR_LH_2P = os.path.join(LABSERVER_DIR_LH, "_2p")
LABSERVER_DIR_LH_BEH = os.path.join(LABSERVER_DIR_LH, "_behavior")
LABSERVER_DIR_JB  = os.path.join(LABSERVER_DIR, "BRAUN_Jonas", "Experimental_data", "2p")
NAS_DIR_JB = os.path.join(NAS_DIR, "JB")
NAS2_DIR_JB = os.path.join(NAS2_DIR, "JB")
NAS_DIR_LH = os.path.join(NAS_DIR, "LH")
NAS2_DIR_LH = os.path.join(NAS2_DIR, "LH")

HOME_DIR = str(Path.home())
TMP_PROCESS_DIR = os.path.join(HOME_DIR, "tmp")
LOCAL_DATA_DIR = os.path.join(HOME_DIR, "data")
LOCAL_DATA_DIR_LONGTERM = os.path.join(LOCAL_DATA_DIR, "longterm")
TWOP_FOLDER = "2p"
PROCESSED_FOLDER = "processed"

RAW_GREEN_TIFF = "green.tif"
RAW_RED_TIFF = "red.tif"

def copy_remote_to_local(trial_dirs, target_base_dir=LOCAL_DATA_DIR, source_base_dir=LABSERVER_DIR_LH_2P, raw=True, xml=True, sync_dirs=None, beh_dirs=None):
    if not isinstance(trial_dirs, list):
        trial_dirs = [trial_dirs]
    if sync_dirs is not None:
        raise NotImplementedError
        if not isinstance(sync_dirs, list):
            sync_dirs = [sync_dirs]
        assert len(sync_dirs) == len(trial_dirs)
    if beh_dirs is not None:
        raise NotImplementedError
        if not isinstance(beh_dirs, list):
            beh_dirs = [beh_dirs]
        assert len(beh_dirs) == len(trial_dirs)

    for i_trial, source_trial_dir in enumerate(trial_dirs):
        source_abs_dir = os.path.join(source_base_dir, source_trial_dir)
        if not os.path.exists(source_abs_dir):
            raise FileNotFoundError
        target_trial_dir = os.path.join(target_base_dir, source_trial_dir)
        if not os.path.exists(target_trial_dir):
            os.makedirs(target_trial_dir)

        target_2p_dir = os.path.join(target_trial_dir, TWOP_FOLDER)
        if raw:
            if not os.path.exists(target_2p_dir):
                os.makedirs(target_2p_dir)
            raw_dir = utils2p.find_raw_file(source_abs_dir)
            _, file_name = os.path.split(raw_dir)
            copyfile(raw_dir, os.path.join(target_2p_dir, file_name))
        
        if xml:
            if not os.path.exists(target_2p_dir):
                os.makedirs(target_2p_dir)
            xml_dir = utils2p.find_metadata_file(source_abs_dir)
            _, file_name = os.path.split(xml_dir)
            copyfile(xml_dir, os.path.join(target_2p_dir, file_name))

def get_flies_from_datedir(date_dir, endswith="", contains=""):
    """get fly directories from date directory.
    Finds all sub-folders start with "Fly" or "fly.

    Parameters
    ----------
    date_dir : str
        base directory where to search

    endswith : str, optional
        constrain search to folders ending with this string, by default ""

    contains : str, optional
        constrain search to folders containing this string, by default ""

    Returns
    -------
    list
        list of sub-directories fullfilling the specified criteria
    """
    dir_list = os.listdir(date_dir)
    # return every subfolder that starts with 'Fly' and ends with user specified argument
    fly_dirs = [os.path.join(date_dir, folder) for folder in dir_list
                    if not os.path.isfile(os.path.join(date_dir, folder))
                    and (folder.startswith('Fly') or folder.startswith('fly'))
                    and folder.endswith(endswith)
                    and contains in folder
               ]
    return sorted(fly_dirs)

def get_trials_from_fly(fly_dir, startswith="", endswith="", contains="", exclude="processed"):
    """get all sub-directories of a fly directory that are trial directories.
    By default, excludes the "processed" folder.
    Attention: this function returns a list of a lists even if only one fly is supplied.

    Parameters
    ----------
    fly_dir : str or list of str
        directory that holds the data for one fly or list containing fly_dir of multiple flies

    startswith : str, optional
        only select folders that start with this string, by default ""

    endswith : str, optional
        only select folders that end with this string, by default ""

    contains : str, optional
        only select folders that contain this string, by default ""

    exclude : str, optional
        exclude folders that contain this string, by default "processed"

    Returns
    -------
    all_trial_dirs: list of lists (even when only one fly_dir supplied)
        one list per fly, each list contains the trial sub-directories.
    """
    if not isinstance(fly_dir, list):
        fly_dir = [fly_dir]

    dir_list = [os.listdir(this_dir) for this_dir in fly_dir]
    # return every subfolder that starts with "startswith" and ends with "endswith"
    trial_dirs = [[os.path.join(fly_dir[i_dir], folder) for folder in fly_dir_list 
                            if not os.path.isfile(os.path.join(fly_dir[i_dir], folder))
                            and folder.endswith(endswith)
                            and folder.startswith(startswith)
                            and contains in folder
                            and not exclude in folder
                        ]
                       for i_dir, fly_dir_list in enumerate(dir_list)
                       ]
    return [sorted(this_dir) for this_dir in trial_dirs]

def load_trial(trial_dir):
    """load the .raw data of a trial

    Parameters
    ----------
    trial_dir : str
        folder which contains the .raw file in one of its sub-folders

    Returns
    -------
    green: numpy array
        data of the green PMT channel

    red: numpy array or None
        data of the red PMT channel. if PMT gain was 0, return None
    """
    trial_xml = utils2p.find_metadata_file(trial_dir)
    trial_raw = utils2p.find_raw_file(trial_dir)

    meta_data = utils2p.Metadata(trial_xml)
    if meta_data.get_gain_b() == 0:
        green = utils2p.load_raw(path=trial_raw, metadata=meta_data)
        return (green[0], None)
    else:
        green, red = utils2p.load_raw(path=trial_raw, metadata=meta_data)
        return (green, red)
    
def convert_raw_to_tiff(trial_dir, overwrite=False, return_stacks=True, green_dir=None, red_dir=None):
    """load .raw files from two-photon microscope and save them as tifs.

    Parameters
    ----------
    trial_dir : str
        folder which contains the .raw file in one of its sub-folders

    overwrite : bool, optional
        whether to overwrite existing files, by default False

    return_stacks : bool, optional
        if True, return numpy array, if not, by default True

    green_dir : str, optional
        where to store the green data. If not supplied, will store in trial_dir/processed/green.tif,
        by default None

    red_dir : str, optional
        where to store the red data. If not supplied, will store in trial_dir/processed/red.tif,
        by default None

    Returns
    -------
    (green: numpy array)
        data from green PMT channel. only if return_stacks == True
    (red: numpy array)
        data from red PMT channel. only if return_stacks == True. None if PMT gain == 0

    Raises
    ------
    NotImplementedError
        if more than two stacks are included in the raw file
    """
    #TODO: make this processed_dir more flexible
    if green_dir is None or red_dir is None:
        processed_dir = os.path.join(trial_dir, PROCESSED_FOLDER)
        if not os.path.exists(os.path.join(processed_dir)):
            os.makedirs(os.path.join(processed_dir))

    green_dir = os.path.join(processed_dir, RAW_GREEN_TIFF) if green_dir is None else green_dir
    red_dir = os.path.join(processed_dir, RAW_RED_TIFF) if red_dir is None else red_dir

    if os.path.isfile(green_dir) and (os.path.isfile(red_dir) or red_dir is None) and not overwrite:
        if not return_stacks:
            return None, None
        green = utils2p.load_img(green_dir)
        try:
            red = utils2p.load_img(red_dir)
        except FileNotFoundError:
            red = None
            Warning("No red tif was found. Returning None. If you recorded it and want to create it, toggle the overwrite Flag")
        return green, red

    stacks = load_trial(trial_dir)
    if len(stacks) == 1:
        green = stacks[0]
        red = None
        utils2p.save_img(green_dir, green)
    elif len(stacks) == 2:
        green, red = stacks
        utils2p.save_img(green_dir, green)
        if red is not None:
            utils2p.save_img(red_dir, red)
    else:
        raise NotImplementedError("More than two stacks are not implemented in load_experiment.")

    return (green, red) if return_stacks else (None, None)
