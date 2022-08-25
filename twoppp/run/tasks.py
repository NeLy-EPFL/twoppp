import os
import time
import shutil
from copy import deepcopy
import datetime
import numpy as np

from typing import Union, Any

from twoppp import load, utils, MODULE_PATH, TWOPPP_PATH
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp.behaviour.fictrac import config_and_run_fictrac
from twoppp.behaviour.stimulation import get_sync_signals_stimulation
from twoppp.rois import prepare_roi_selection
from twoppp.plot import show3d
from twoppp.run.runparams import global_params, CURRENT_USER
from twoppp.run.runutils import get_selected_trials, get_scratch_fly_dict, find_trials_2plinux, send_email

class Task:
    def __init__(self, prio: int=0) -> None:
        self.prio = prio
        self.name = ""
        self.params = None
        self.previous_tasks = []
        self.t_wait_s = 0
        self.send_status_emails = CURRENT_USER["send_emails"]

    def test_todo(self, fly_dict: dict, file_name: str) -> bool:
        TODO = False
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = []
            if any([isinstance(task, BehDataTransferTask) for task in self.previous_tasks]):
                twoplinux_trial_names += find_trials_2plinux(fly_dict["dir"], CURRENT_USER["initials"], twop=False)
            if any([isinstance(task, TwopDataTransferTask) for task in self.previous_tasks]) or \
                any([isinstance(task, SyncDataTransfer) for task in self.previous_tasks]):
                twoplinux_trial_names += find_trials_2plinux(fly_dict["dir"], CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        files = [os.path.join(trial_dir, load.PROCESSED_FOLDER, file_name) for trial_dir in trial_dirs_todo]

        if not all([os.path.isfile(this_file) for this_file in files]):
            TODO = True

        return TODO

    def send_start_email(self, fly_dict: dict) -> None:
        if self.send_status_emails:
            try:
                status = fly_dict["status"].upper()
                fly_name = " ".join(fly_dict["dir"].split(os.sep)[-2:])
                subject = f"{status}: {fly_dict['tasks']} {fly_name}"
                message = f"{fly_dict['dir']} \n{fly_dict['selected_trials']} \n{fly_dict['args']}"
                send_email(subject, message, receiver_email=CURRENT_USER["email"])
            except Exception as e:
                print("Error while sending status mail. Will proceed with processing.")
                print(e)

    def run(self, fly_dict: dict, params: PreProcessParams = None) -> bool:
        raise NotImplementedError

    def run_multiple_flies(self, fly_dicts: list, params: PreProcessParams = None) -> bool:
        raise NotImplementedError

    def test_previous_task_ready(self, fly_dict: dict) -> bool:
        """returns True if previous task is ready and False otherwise

        Parameters
        ----------
        fly_dict : dict
            [description]

        Returns
        -------
        bool
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        if isinstance(self.previous_tasks, list):
            return not any([task.test_todo(fly_dict) for task in self.previous_tasks])
        else:
            raise NotImplementedError

    def wait_for_previous_task(self, fly_dict: dict) -> bool:
        """returns True if previous task is ready.
        If previous task is not ready, waits for self.t_wait_s and checks again whether previous task is ready
        if previous task is still not ready, return False. If ready, returns True

        Parameters
        ----------
        fly_dict : dict
            [description]

        Returns
        -------
        bool
            [description]
        """
        if self.t_wait_s:
            if not self.test_previous_task_ready(fly_dict):
                names = [task.name for task in self.previous_tasks]
                print(f"Waiting for tasks {names} of fly {fly_dict['dir']} to finish.")
                time.sleep(self.t_wait_s)
                return self.test_previous_task_ready(fly_dict)
            return True
        else:
            return self.test_previous_task_ready(fly_dict)

class TwopDataTransferTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "twop_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"], CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "Image_0001_0001.raw", raise_error=False) is None:
                return True
            if utils.find_file(trial_dir, "Experiment.xml", raise_error=False) is None:
                return True
        return False

class SyncDataTransfer(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "sync_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"], CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "Episode001.h5", raise_error=False) is None:
                return True
            if utils.find_file(trial_dir, "ThorRealTimeDataSettings.xml", raise_error=False) is None:
                return True
        return False

class BehDataTransferTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "beh_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"], CURRENT_USER["initials"], twop=False)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "capture_metadata.json", raise_error=False) is None:
                return True
            if utils.find_file(trial_dir, f"camera_{CURRENT_USER['fictrac_cam']}.mp4", raise_error=False) is None:  # TODO: make more general
                return True
        return False

class TifTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "tif"
        self.previous_tasks = [TwopDataTransferTask()]

    def test_todo(self, fly_dict):
        return super().test_todo(fly_dict, file_name=global_params.green_raw)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.twoway_align = False
        self.params.ref_frame = ""  # -> don't save ref frame
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        # convert raw to tiff
        preprocess.run_all_trials()
        return True

class PreClusterTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "pre_cluster"
        self.previous_tasks = [TwopDataTransferTask()]

    def test_todo(self, fly_dict):
        TODO1 = super().test_todo(fly_dict, file_name=global_params.red_com_crop)
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        if not os.path.isdir(scratch_fly_dict["dir"]) or len(os.listdir(scratch_fly_dict["dir"])) == 0:
            TODO2 = True
        else:
            TODO2 = super().test_todo(scratch_fly_dict, file_name=global_params.red_com_crop) if not CURRENT_USER["ignore_scratch"] else False
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.twoway_align = True
        self.params.use_com = True
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        # convert raw to tiff and perform COM registration
        preprocess.run_all_trials()
        
        print("COPYING TO CLUSTER: ", fly_dict["dir"])
        utils.run_shell_command(". " + os.path.join(TWOPPP_PATH, "register",
                                                    "copy_to_cluster.sh") + " " + fly_dict["dir"])
        
        return True

class ClusterTask(Task):
    def __init__(self, prio=0, N_per_batch=28):
        super().__init__(prio)
        self.name = "cluster"
        self.previous_tasks = [PreClusterTask()]
        self.N_per_batch = N_per_batch
        self.t_sleep_s = 60

    def test_todo(self, fly_dict):
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        if not os.path.isdir(scratch_fly_dict["dir"]) or len(os.listdir(scratch_fly_dict["dir"])) == 0:
            return True
        else:
            return super().test_todo(scratch_fly_dict, file_name=global_params.red_com_warped) if not CURRENT_USER["ignore_scratch"] else False

    def run(self, fly_dict, params):
        if not self.wait_for_previous_task(fly_dict):
            return False

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        scratch_trial_dirs = get_selected_trials(scratch_fly_dict)
        red_com_crop = utils.get_stack(os.path.join(scratch_trial_dirs[0], load.PROCESSED_FOLDER, self.params.red_com_crop))
        N_frames = red_com_crop.shape[0]
        del red_com_crop
        N_batches = np.ceil(N_frames/self.N_per_batch).astype(int)

        status_files = np.zeros(len(scratch_trial_dirs)).astype(int)
        status_file_name_split = self.params.motion_field.split(".")
        file_base = status_file_name_split[0] + "_"
        file_end = "." + status_file_name_split[1]

        for i_trial, trial_dir in enumerate(scratch_trial_dirs):
            for i_file in np.arange(status_files[i_trial], N_batches):
                if os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER, file_base+f"{i_file}"+file_end)):
                    status_files[i_trial] = i_file + 1
                else:
                    continue
    
        print(time.ctime(time.time()), scratch_fly_dict["dir"], "cluster status:", status_files, f"/{N_batches}")
        if any(np.array(status_files) < N_batches):
            time.sleep(self.t_sleep_s)
            return False  # cluster is not yet finished
        else:
            return True  # cluster is finished. Next task can begin

    def run_multiple_flies(self, fly_dicts, params):
        if not all([self.wait_for_previous_task(fly_dict) for fly_dict in fly_dicts]):
            return False
        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        scratch_trial_dirs = []
        for fly_dict in fly_dicts:
            scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
            scratch_trial_dirs += get_selected_trials(scratch_fly_dict)
        red_com_crop = utils.get_stack(os.path.join(scratch_trial_dirs[0], load.PROCESSED_FOLDER, self.params.red_com_crop))
        N_frames = red_com_crop.shape[0]
        del red_com_crop
        N_batches = np.ceil(N_frames/self.N_per_batch).astype(int)

        status_files = np.zeros(len(scratch_trial_dirs)).astype(int)
        status_file_name_split = self.params.motion_field.split(".")
        file_base = status_file_name_split[0] + "_"
        file_end = "." + status_file_name_split[1]

        def chop_microseconds(delta):
            return delta - datetime.timedelta(microseconds=delta.microseconds)

        N_rep = 0
        while self.test_todo(fly_dict):
            for i_trial, trial_dir in enumerate(scratch_trial_dirs):
                for i_file in np.arange(status_files[i_trial], N_batches):
                    if os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER, file_base+f"{i_file}"+file_end)):
                        status_files[i_trial] = i_file + 1
                    else:
                        continue
            # status_perc = 100 * status_files / N_batches
            print(time.ctime(time.time()), status_files, f"/{N_batches}")
            if N_rep == 0:
                start_time = time.time()
                status_start = deepcopy(status_files)
            else:
                diff_N = status_files - status_start
                diff_t = time.time() - start_time
                speed = diff_N / diff_t
                speed[speed == 0] = -999999999
                rem_N = N_batches - status_files
                rem_t = rem_N / speed
                print(f"remaining time: {[str(chop_microseconds(datetime.timedelta(seconds=t))) for t in rem_t]}")
            N_rep += 1
            time.sleep(self.t_sleep_s)
        
        return True

class PostClusterTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "post_cluster"
        self.previous_tasks = [ClusterTask()]

    def test_todo(self, fly_dict):
        return super().test_todo(fly_dict, file_name=global_params.green_com_warped)
    
    def run(self, fly_dict, params=None):
        # wait_for_previous_task
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        # use this parameter combination to only run warping after successful com registration
        # this makes sure that the "overwrite" only applies to the warping and not to com
        self.params.use_com = False
        self.params.red_raw = self.params.red_com_crop
        self.params.green_raw = self.params.green_com_crop
        self.params.use_warp = True
        self.params.overwrite = fly_dict["overwrite"]


        print("COPYING BACK FLY: ", fly_dict["dir"])
        utils.run_shell_command(". " + os.path.join(TWOPPP_PATH, "register",
                                                    "copy_from_cluster.sh") + " " + fly_dict["dir"])

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        # finalise motion correction
        _ = [preprocess._warp_trial(processed_dir) for processed_dir in preprocess.trial_processed_dirs]
        # preprocess.run_all_trials()
        return True

class DenoiseTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "denoise"
        self.previous_tasks = [PostClusterTask()]
    
    def test_todo(self, fly_dict):
        return super().test_todo(fly_dict, file_name=global_params.green_denoised)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.use_com = True
        self.params.use_warp = True
        self.params.use_denoise = True
        self.params.overwrite = fly_dict["overwrite"]

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        # finalise motion correction, denoise, compute dff
        preprocess._denoise_all_trials()
        # preprocess.run_all_trials()
        return True

class DffTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "dff"
        self.previous_tasks = [DenoiseTask()]

    def test_todo(self, fly_dict):
        return super().test_todo(fly_dict, file_name=global_params.dff)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.use_com = True
        self.params.use_warp = True
        self.params.use_denoise = True
        self.params.use_dff = True
        self.params.make_summary_stats = False
        self.params.overwrite = fly_dict["overwrite"]

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        preprocess._compute_dff_alltrials()
        # preprocess.run_all_trials()
        return True

class SummaryStatsTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "summary_stats"
        self.previous_tasks = [DffTask()]

    def test_todo(self, fly_dict):
        return not os.path.isfile(os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, global_params.summary_stats))

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.use_com = True
        self.params.use_warp = True
        self.params.use_denoise = True
        self.params.use_dff = True
        self.params.make_summary_stats = True
        self.params.overwrite = fly_dict["overwrite"]

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        preprocess._compute_summary_stats()
        # preprocess.run_all_trials()
        return True

class PrepareROISelectionTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "prepare_roi_selection"
        self.previous_tasks = [SummaryStatsTask()]

    def test_todo(self, fly_dict):
        TODO1 = not os.path.isfile(os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, global_params.pca_maps))
        TODO2 = not os.path.isfile(os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, global_params.pca_maps_plot))
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        prepare_roi_selection(fly_dict["dir"], trial_dirs, self.params)

        return True

class ROISelectionTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "roi_selection"
        self.previous_tasks = [PrepareROISelectionTask()]

    def test_todo(self, fly_dict):
        return not os.path.isfile(os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, global_params.roi_centers))

    def run(self,fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            print("You'll have to do the ROI detection manually.")
            return False

class ROISignalsTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "roi_signals"
        self.previous_tasks = [ROISelectionTask()]

    def test_todo(self, fly_dict):
        return not os.path.isfile(os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, global_params.roi_mask))

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        preprocess.extract_rois()
        return True

    

class FictracTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "fictrac"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        # print("TODO: implement FictracTask.test_todo() method!!!")
        TODO1 = super().test_todo(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        TODO2 = not all([bool(len(utils.find_file(os.path.join(trial_dir, "behData", "images"), name=f"camera_{CURRENT_USER['fictrac_cam']}-*.dat", raise_error=False))) for trial_dir in trial_dirs_todo])
        return TODO1  or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)
        # this has no inbuilt override protection --> only protected by the test_todo() method of this Task
        config_and_run_fictrac(fly_dict["dir"], trial_dirs)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        preprocess.get_dfs()
        return True

    def run_multiple_flies(self, fly_dicts, params=None):
        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        if isinstance(self.params, list):
            N_params = len(self.params)
        else:
            N_params = 1
        all_selected_trials = []
        for fly_dict in fly_dicts:
            all_selected_trials += get_selected_trials(fly_dict)
        
        config_and_run_fictrac("multiple flies", all_selected_trials)

        for i_fly, fly_dict in enumerate(fly_dicts):
            print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
            params = self.params if N_params == 1 else self.params[i_fly]
            trial_dirs = get_selected_trials(fly_dict)
            preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=params, trial_dirs=trial_dirs)
            preprocess.get_dfs()

class Df3dTask(Task):
    def __init__(self, prio: int=0) -> None:
        super().__init__(prio)
        self.name = "df3d"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict: dict) -> bool:
        TODO1 = super().test_todo(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        TODO2 = not all([os.path.isdir(os.path.join(trial_dir, "behData", "images", "df3d")) for trial_dir in trial_dirs_todo])
        if not TODO2:
            TODO2 = not all([bool(len(utils.find_file(os.path.join(trial_dir, "behData", "images", "df3d"), name=f"aligned_pose__*.pkl", raise_error=False))) for trial_dir in trial_dirs_todo])
        return TODO1  or TODO2

    def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        preprocess._pose_estimate()
        preprocess._post_process_pose()

        return True

class VideoTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "video"
        self.previous_tasks = [DffTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        trial_dirs = get_selected_trials(fly_dict)
        TODO = False
        for trial_dir in trial_dirs:
            trial_name = "_".join(trial_dir.split(os.sep)[-3:])
            file_name = trial_name+"_"+global_params.dff_beh_video_name+".mp4"
            TODO = TODO or not os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER, file_name))
            TODO = TODO or not os.path.isfile(os.path.join(CURRENT_USER["video_dir"], file_name))
        return TODO
    
    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.dff_video_max_length = None # 100
        self.params.dff_video_downsample = 2
        self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)
        for i, trial_dir in enumerate(trial_dirs):
            preprocess._make_dff_behaviour_video_trial(i_trial=i, mask=None, include_2p=True)
            shutil.copy2(
                os.path.join(trial_dir, load.PROCESSED_FOLDER, f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_{preprocess.trial_names[i]}_{preprocess.params.dff_beh_video_name}.mp4"), 
                CURRENT_USER["video_dir"])

        return True

class LaserStimProcessTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "stim_process"
        self.previous_tasks = [FictracTask()]

    def test_todo(self, fly_dict):
        TODO1 = super().test_todo(fly_dict, file_name="stim_paradigm.pkl")
        TODO2 = super().test_todo(fly_dict, file_name=global_params.df3d_df_out_dir)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        for trial_dir in trial_dirs:
            beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, self.params.df3d_df_out_dir)
            _ = get_sync_signals_stimulation(trial_dir,
                                             sync_out_file="stim_sync.pkl",
                                             paradigm_out_file="stim_paradigm.pkl",
                                             overwrite=self.params.overwrite,
                                             index_df=beh_df,
                                             df_out_dir=beh_df)
        return True

class VideoPlot3DTask(Task):
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "video_plot_3d"
        self.previous_tasks = [TifTask()]

    def test_todo(self, fly_dict):
        TODO1 = super().test_todo(fly_dict, file_name="zstack.mp4") or \
            super().test_todo(fly_dict, file_name="ystack.mp4") or \
            super().test_todo(fly_dict, file_name="xstack.mp4")
        TODO2 = super().test_todo(fly_dict, file_name="3dproject.png")
        return True  # TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_start_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params, trial_dirs=trial_dirs)

        for i, processed_dir in enumerate(preprocess.trial_processed_dirs):
            
            red_tif = os.path.join(processed_dir, self.params.red_raw)
            show3d.make_avg_videos_3d(
                green=os.path.join(processed_dir, self.params.green_raw),
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=os.path.join(processed_dir, "green_avg.tif"),
                red_avg=os.path.join(processed_dir, "red_avg.tif") if os.path.isfile(red_tif) else None,
                out_dir=processed_dir
            )
            show3d.plot_projections_3d(
                green=os.path.join(processed_dir, self.params.green_raw),
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=os.path.join(processed_dir, "green_avg.tif"),
                red_avg=os.path.join(processed_dir, "red_avg.tif") if os.path.isfile(red_tif) else None,
                out_dir=processed_dir
            )
            
            shutil.copy2(
                os.path.join(processed_dir, "zstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"], f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_{preprocess.trial_names[i]}_zstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "ystack.mp4"),
                os.path.join(CURRENT_USER["video_dir"], f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_{preprocess.trial_names[i]}_ystack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "xstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"], f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_{preprocess.trial_names[i]}_xstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "3dproject.png"),
                os.path.join(CURRENT_USER["video_dir"], f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_{preprocess.trial_names[i]}_3dproject.png")
            )

        return True

task_collection = {
    "twop_data_transfer": TwopDataTransferTask(prio=-100),
    "beh_data_transfer": BehDataTransferTask(prio=-100),
    "pre_cluster": PreClusterTask(prio=10),
    "tif": TifTask(prio=5),
    "cluster": ClusterTask(prio=-100),
    "post_cluster": PostClusterTask(prio=-5),
    "denoise": DenoiseTask(prio=-6),
    "dff": DffTask(prio=-10),
    "summary_stats": SummaryStatsTask(prio=-16),
    "prepare_roi_selection": PrepareROISelectionTask(prio=-17),
    "roi_selection": ROISelectionTask(prio=-100),
    "roi_signals": ROISignalsTask(prio=-18),
    "fictrac": FictracTask(prio=0),
    "df3d": Df3dTask(prio=-20),
    "video": VideoTask(prio=-15),
    "laser_stim_process": LaserStimProcessTask(prio=-1),
    "video_plot_3d": VideoPlot3DTask(prio=-15)
}
