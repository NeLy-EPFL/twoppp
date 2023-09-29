"""
sub-module to define different steps of pre-processing as sub-classes of Task
"""
import os
import time
import shutil
from copy import deepcopy
import datetime
from typing import List
import numpy as np

from twoppp import load, utils, TWOPPP_PATH
from twoppp.register import warping
from twoppp.pipeline import PreProcessFly, PreProcessParams
from twoppp.behaviour.fictrac import config_and_run_fictrac
from twoppp.behaviour.stimulation import get_sync_signals_stimulation
from twoppp.rois import prepare_roi_selection
from twoppp.plot import show3d
from twoppp.run.runparams import global_params, CURRENT_USER
from twoppp.run.runutils import get_selected_trials, get_scratch_fly_dict, find_trials_2plinux
from twoppp.run.runutils import send_email

class Task:
    """
    Base class to implement a particular pre-processing step.
    """
    def __init__(self, prio: int=0) -> None:
        """
        Base class to implement a particular pre-processing step.

        Parameters
        ----------
        prio : int, optional
            priority of this particular pre-processing step, by default 0
        """
        self.prio = prio
        self.name = ""
        self.params = None
        self.previous_tasks = []
        self.t_wait_s = 0
        self.send_status_emails = CURRENT_USER["send_emails"]

    def _test_todo_trials(self, fly_dict: dict, file_name: str) -> bool:
        """
        abstract method to be re-used in test_todo()
        Function check whether the current task still needs to be performed or is already finished
        by checking the existence of a particular file in each trial's processed sub-folder.

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments
        file_name : str
            name of the file to be searched for in each trial processed folder

        Returns
        -------
        bool
            True if at least one trial has to be done.
            False, if all processing for this task is finished.
        """
        TODO = False
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = []
            if any([isinstance(task, BehDataTransferTask) for task in self.previous_tasks]):
                twoplinux_trial_names += find_trials_2plinux(fly_dict["dir"],
                                                             CURRENT_USER["initials"],
                                                             twop=False)
            if any([isinstance(task, TwopDataTransferTask) for task in self.previous_tasks]) or \
                any([isinstance(task, SyncDataTransfer) for task in self.previous_tasks]):
                twoplinux_trial_names += find_trials_2plinux(fly_dict["dir"],
                                                             CURRENT_USER["initials"],
                                                             twop=True)
        else:
            twoplinux_trial_names = None
        try:
            trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)
        except:
            TODO = True
            return TODO
            
        files = [os.path.join(trial_dir, load.PROCESSED_FOLDER, file_name)
                 for trial_dir in trial_dirs_todo]

        if not all([os.path.isfile(this_file) for this_file in files]):
            TODO = True

        return TODO

    def _test_todo_fly(self, fly_dict: dict, file_name: str) -> bool:
        """
        abstract method to be re-used in test_todo()
        Function check whether the current task still needs to be performed or is already finished
        by checking the existence of a particular file in the fly directory's processed sub-folder.

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments
        file_name : str
            name of the file to be searched for in the fly processed folder

        Returns
        -------
        bool
            True if still has to be done.
            False, if all processing for this task is finished.
        """
        this_file = os.path.join(fly_dict["dir"], load.PROCESSED_FOLDER, file_name)
        return not os.path.isfile(this_file)

    def test_todo(self, fly_dict: dict) -> bool:
        """
        check whether this task still needs to be performed for a particular fly.
        Implement this in every sub-class.
        You can use the _test_todo_trials and _test_todo_fly methods.

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments

        Returns
        -------
        bool
            True if still to do, False if already completed.
        """
        raise NotImplementedError

    def send_status_email(self, fly_dict: dict) -> None:
        """
        Send an e-mail to indicate that the processing of the task has started for a fly.

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments
        """
        if self.send_status_emails:
            try:
                status = fly_dict["status"].upper()
                fly_name = " ".join(fly_dict["dir"].split(os.sep)[-2:])
                subject = f"{status}: {fly_dict['tasks']} {fly_name}"
                message = f"{fly_dict['dir']} \n{fly_dict['selected_trials']} \n{fly_dict['args']}"
                send_email(subject, message, receiver_email=CURRENT_USER["email"])
            except Exception as error:
                print("Error while sending status mail. Will proceed with processing.")
                print(error)

    def run(self, fly_dict: dict, params: PreProcessParams = None) -> bool:
        """
        run the current task on a particular fly.

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments
        params : PreProcessParams, optional
            instance of class PreProcessParams, by default None

        Returns
        -------
        bool
            success: True if finished, False if waiting for previous task
        """
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            return True

    def run_multiple_flies(self, fly_dicts: List[dict], params: PreProcessParams = None) -> bool:
        """
        Possible future functionality: run one processing step on multiple flies at the same time.
        This might be usefull when a processing step can be parallelised, e.g. for fictrac.
        For now, this is not yet used

        Parameters
        ----------
        fly_dicts : List[dict]
            list of dictionaries, each containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments
        params : PreProcessParams, optional
            instance of class PreProcessParams, by default None

        Returns
        -------
        bool
            success: True if finished, False if waiting for previous task
        """
        return False

    def test_previous_task_ready(self, fly_dict: dict) -> bool:
        """
        returns True if previous tasks are finished and False otherwise

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments

        Returns
        -------
        bool
            True if all previous tasks have finished, False otherwise

        Raises
        ------
        NotImplementedError
            if not isinstance(self.previous_tasks, list)
        """
        if isinstance(self.previous_tasks, list):
            return not any([task.test_todo(fly_dict) for task in self.previous_tasks])
        else:
            raise NotImplementedError

    def wait_for_previous_task(self, fly_dict: dict) -> bool:
        """
        returns True if all previous tasks have finished.
        If previous task is not finished, waits for self.t_wait_s
        and checks again whether previous task is finished
        if previous task is still not finished, return False. If finished, returns True

        Parameters
        ----------
        fly_dict : dict
            dictionary containing the following information:
            - dir: str, the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: bool, whether or not to force an overwrite of the previous results
            - status: bool, whether the todo is "ready","running", "done", or "waiting"
            - args: list[str], additional arguments

        Returns
        -------
        bool
            True if all previous tasks have finished, False otherwise
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
    """
    Dummy Task to check whether twop photon data was already transfered
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "twop_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=True)
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
    """
    Dummy Task to check whether ThorSync data was already transfered
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "sync_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=True)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "Episode001.h5", raise_error=False) is None:
                return True
            if utils.find_file(trial_dir, "ThorRealTimeDataSettings.xml",raise_error=False) is None:
                return True
        return False

class BehDataTransferTask(Task):
    """
    Dummy Task to check whether behavioural data was already transfered
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "beh_data_transfer"

    def test_todo(self, fly_dict):
        if CURRENT_USER["check_2plinux_trials"]:
            twoplinux_trial_names = find_trials_2plinux(fly_dict["dir"],
                                                        CURRENT_USER["initials"], twop=False)
        else:
            twoplinux_trial_names = None
        trial_dirs_todo = get_selected_trials(fly_dict, twoplinux_trial_names)

        for trial_dir in trial_dirs_todo:
            if utils.find_file(trial_dir, "capture_metadata.json", raise_error=False) is None:
                return True
            cam_name = f"camera_{CURRENT_USER['fictrac_cam']}.mp4"
            if utils.find_file(trial_dir, cam_name, raise_error=False) is None:
                return True
        return False

class TifTask(Task):
    """
    Task to convert .raw files to .tif files
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "tif"
        self.previous_tasks = [TwopDataTransferTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.green_raw)
        TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.red_raw)
        return TODO1 and TODO2  # only do this task if both are missing

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        #self.params.twoway_align = False
        #self.params.ref_frame = ""  # -> don't save ref frame
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # convert raw to tiff
        preprocess.run_all_trials()
        return True

class PreClusterTask(Task):
    """
    Task to convert .raw to tif, perform center of mass registration and copy necessay files
    for optic flow motion correction to the cluster
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "pre_cluster"
        self.previous_tasks = [TwopDataTransferTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.red_com_crop)
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        if not os.path.isdir(scratch_fly_dict["dir"]) or \
            len(os.listdir(scratch_fly_dict["dir"])) == 0:
            TODO2 = True
        elif CURRENT_USER["ignore_scratch"]:
            TODO2 = False
        else:
            TODO2 = self._test_todo_trials(scratch_fly_dict, file_name=global_params.red_com_crop)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        #self.params.twoway_align = True
        #self.params.use_com = True
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # convert raw to tiff and perform COM registration
        preprocess.run_all_trials()

        print("COPYING TO CLUSTER: ", fly_dict["dir"])
        utils.run_shell_command(os.path.join(TWOPPP_PATH, "register",
                                                    "copy_to_cluster.sh") + " " + fly_dict["dir"])

        fly_dict["status"] = "done"
        self.send_status_email(fly_dict)
        return True


class PreClusterGreenOnlyTask(Task):
    """
    Task to convert .raw to tif, perform center of mass registration and copy necessary files
    for optic flow motion correction to the cluster.
    Perform all of that on the green channel.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "pre_cluster_green_only"
        self.previous_tasks = [TifTask()]  # TwopDataTransferTask()
        self.params = deepcopy(global_params)
        self.params.green_raw = "green.tif"
        self.params.red_raw = "green.tif"
        self.params.ref_frame_com = "ref_frame_com.tif"
        self.params.green_com_crop = "green_com_crop.tif"
        self.params.red_com_crop = "red_com_crop.tif"
        self.params.green_com_warped = "green_com_warped.tif"
        self.params.red_com_warped = "red_com_warped.tif"

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name=self.params.red_com_crop)
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        if not os.path.isdir(scratch_fly_dict["dir"]) or \
            len(os.listdir(scratch_fly_dict["dir"])) == 0:
            TODO2 = True
        elif CURRENT_USER["ignore_scratch"]:
            TODO2 = False
        else:
            TODO2 = self._test_todo_trials(scratch_fly_dict, file_name=self.params.red_com_crop)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        # self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        #self.params.twoway_align = True
        self.params.use_com = True
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # convert raw to tiff and perform COM registration
        # preprocess.run_all_trials()
        # preprocess._save_ref_frame()
        preprocess.ref_frame = os.path.join(preprocess.fly_processed_dir, preprocess.params.ref_frame)
        ref_stack = os.path.join(preprocess.trial_processed_dirs[preprocess.params.i_ref_trial], preprocess.params.red_raw)
        warping.save_ref_frame(stack=ref_stack,
                               ref_frame_dir=preprocess.ref_frame,
                               i_frame=preprocess.params.i_ref_frame,
                               com_pre_reg=preprocess.params.use_com,
                               overwrite=preprocess.params.overwrite,
                               crop=preprocess.params.post_com_crop_values
                               if preprocess.params.post_com_crop else None)
        _ = [preprocess._com_correct_trial(processed_dir) for processed_dir in preprocess.trial_processed_dirs]

        print("COPYING TO CLUSTER: ", fly_dict["dir"])
        utils.run_shell_command(". " + os.path.join(TWOPPP_PATH, "register",
                                                    "copy_to_cluster.sh") + " " + fly_dict["dir"])

        fly_dict["status"] = "done"
        self.send_status_email(fly_dict)
        return True

class ClusterTask(Task):
    """
    Task to check the current status of processing on the cluster.
    Will print progress of cluster.
    To start the actual processing on the cluster, follow the instructions in the README.md
    """
    def __init__(self, prio=0, n_per_batch=28):
        super().__init__(prio)
        self.name = "cluster"
        self.previous_tasks = [PreClusterTask()]
        self.n_per_batch = n_per_batch
        self.t_sleep_s = 60

    def test_todo(self, fly_dict):
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        if not os.path.isdir(scratch_fly_dict["dir"]) or \
            len(os.listdir(scratch_fly_dict["dir"])) == 0:
            return True
        elif CURRENT_USER["ignore_scratch"]:
            return False
        else:
            return self._test_todo_trials(scratch_fly_dict, file_name=global_params.red_com_warped)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        scratch_fly_dict = get_scratch_fly_dict(fly_dict, scratch_base=CURRENT_USER["scratch_dir"])
        scratch_trial_dirs = get_selected_trials(scratch_fly_dict)
        red_com_crop = utils.get_stack(os.path.join(scratch_trial_dirs[0], load.PROCESSED_FOLDER,
                                                    self.params.red_com_crop))
        n_frames = red_com_crop.shape[0]
        del red_com_crop
        n_batches = np.ceil(n_frames/self.n_per_batch).astype(int)

        status_files = np.zeros(len(scratch_trial_dirs)).astype(int)
        status_file_name_split = self.params.motion_field.split(".")
        file_base = status_file_name_split[0] + "_"
        file_end = "." + status_file_name_split[1]

        for i_trial, trial_dir in enumerate(scratch_trial_dirs):
            for i_file in np.arange(status_files[i_trial], n_batches):
                if os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER,
                                  file_base+f"{i_file}"+file_end)):
                    status_files[i_trial] = i_file + 1
                else:
                    continue

        print(time.ctime(time.time()), scratch_fly_dict["dir"],
              "cluster status:", status_files, f"/{n_batches}")
        if any(np.array(status_files) < n_batches):
            time.sleep(self.t_sleep_s)
            return False  # cluster is not yet finished
        else:
            return True  # cluster is finished. Next task can begin

    def run_multiple_flies(self, fly_dicts: List[dict], params: PreProcessParams=None) -> bool:
        if not all([self.wait_for_previous_task(fly_dict) for fly_dict in fly_dicts]):
            return False
        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        scratch_trial_dirs = []
        for fly_dict in fly_dicts:
            scratch_fly_dict = get_scratch_fly_dict(fly_dict,
                                                    scratch_base=CURRENT_USER["scratch_dir"])
            scratch_trial_dirs += get_selected_trials(scratch_fly_dict)
        red_com_crop = utils.get_stack(os.path.join(scratch_trial_dirs[0],
                                                    load.PROCESSED_FOLDER,
                                                    self.params.red_com_crop))
        n_frames = red_com_crop.shape[0]
        del red_com_crop
        n_batches = np.ceil(n_frames/self.n_per_batch).astype(int)

        status_files = np.zeros(len(scratch_trial_dirs)).astype(int)
        status_file_name_split = self.params.motion_field.split(".")
        file_base = status_file_name_split[0] + "_"
        file_end = "." + status_file_name_split[1]

        def chop_microseconds(delta):
            return delta - datetime.timedelta(microseconds=delta.microseconds)

        n_rep = 0
        while any([self.test_todo(fly_dict) for fly_dict in fly_dicts]):
            for i_trial, trial_dir in enumerate(scratch_trial_dirs):
                for i_file in np.arange(status_files[i_trial], n_batches):
                    if os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER,
                                                   file_base+f"{i_file}"+file_end)):
                        status_files[i_trial] = i_file + 1
                    else:
                        continue
            # status_perc = 100 * status_files / n_batches
            print(time.ctime(time.time()), status_files, f"/{n_batches}")
            if n_rep == 0:
                start_time = time.time()
                status_start = deepcopy(status_files)
            else:
                diff_n = status_files - status_start
                diff_t = time.time() - start_time
                speed = diff_n / diff_t
                speed[speed == 0] = -999999999
                rem_n = n_batches - status_files
                rem_t = rem_n / speed
                time_strs = [str(chop_microseconds(datetime.timedelta(seconds=t))) for t in rem_t]
                print(f"remaining time: {time_strs}")
            n_rep += 1
            time.sleep(self.t_sleep_s)

        return True

class PostClusterTask(Task):
    """
    Task to copy data back from the cluster to the local machine and finishing the motion correction
    by applying the motion field to the green channel.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "post_cluster"
        self.previous_tasks = [ClusterTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_trials(fly_dict, file_name=global_params.green_com_warped)

    def run(self, fly_dict, params=None):
        # wait_for_previous_task
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
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
        utils.run_shell_command(os.path.join(TWOPPP_PATH, "register",
                                                    "copy_from_cluster.sh") + " " + fly_dict["dir"])

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # finalise motion correction
        _ = [preprocess._warp_trial(processed_dir)
             for processed_dir in preprocess.trial_processed_dirs]
        # preprocess.run_all_trials()
        return True

class ReplaceMovingFrames(Task):
    """
    Task to replace frames where the social arena is moved while introducing the second fly.
    It replaces the moved frames with the last good frame.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "replace_moving_frames"
        self.previous_tasks = [PreClusterTask()]

    def test_todo(self, fly_dict):####TO DO#######
        return self._test_todo_trials(fly_dict, file_name=global_params.green_denoised)

    def run(self, fly_dict, params=None):
        #if not self.wait_for_previous_task(fly_dict):
        #    return False
        #else:
        #    self.send_status_email(fly_dict)
        #    print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        #self.params.use_com = True
        #self.params.use_warp = True
        #self.params.use_denoise = True
        #self.params.overwrite = fly_dict["overwrite"]

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess._replace_moving_frames()
        return True

class DenoiseTask(Task):
    """
    Task to apply denoising of the green channel by
    training a DeepInterpolation model on the first trial and applying it on all trials.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "denoise"
        self.previous_tasks = [PostClusterTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_trials(fly_dict, file_name=global_params.green_denoised)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.use_com = True
        self.params.use_warp = True
        self.params.use_denoise = True
        self.params.overwrite = fly_dict["overwrite"]

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        # finalise motion correction, denoise, compute dff
        preprocess._denoise_all_trials()
        # preprocess.run_all_trials()
        return True

class DffTask(Task):
    """
    Task to compute the fluorescence baseline for the entire fly and the compute the DFF.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "dff"
        #self.previous_tasks = [PostClusterTask()]
        self.previous_tasks = [DenoiseTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_trials(fly_dict, file_name=global_params.dff)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
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
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess._compute_dff_alltrials()
        # preprocess.run_all_trials()
        return True

class SummaryStatsTask(Task):
    """
    Task to pre-compute summary stats about each trial,
    for exampl the standard deviation, the mean, the maximum,
    the local correlations. Usefull to explore the data and for ROI selection.
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "summary_stats"
        self.previous_tasks = [DffTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_fly(fly_dict, global_params.summary_stats)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
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
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess._compute_summary_stats()
        # preprocess.run_all_trials()
        return True

class PrepareROISelectionTask(Task):
    """
    Task to perform PCA on selected pixels to create PCA maps,
    i.e., pca weights projected into the image space.
    Required to perform before running the ROI selection notebook
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "prepare_roi_selection"
        self.previous_tasks = [SummaryStatsTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_fly(fly_dict, global_params.pca_maps)
        TODO2 = self._test_todo_fly(fly_dict, global_params.pca_maps_plot)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        prepare_roi_selection(fly_dict["dir"], trial_dirs,
                              green_com_warped=self.params.green_com_warped,
                              green_denoised=self.params.green_denoised,
                              summary_stats=self.params.summary_stats,
                              out_file_name=self.params.pca_maps,
                              out_plot_file_name=self.params.pca_maps_plot)

        fly_dict["status"] = "done"
        self.send_status_email(fly_dict)
        return True

class ROISelectionTask(Task):
    """
    Dummy Task for ROI selection.
    Still has to be performed manually, for example using the following notebook:
    manual_ROI_detection_example.ipynb
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "roi_selection"
        self.previous_tasks = [PrepareROISelectionTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_fly(fly_dict, global_params.roi_centers)

    def run(self,fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            print("You'll have to do the ROI detection manually." +\
                  "You can for example use the manual_ROI_detection_example.ipynb notebook.")
            return False

class ROISignalsTask(Task):
    """
    Task to perform signal extraction of ROIs based on the ROI centers defined in ROISelectionTask()
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "roi_signals"
        self.previous_tasks = [ROISelectionTask()]

    def test_todo(self, fly_dict):
        return self._test_todo_fly(fly_dict, global_params.roi_mask)

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess.extract_rois()
        return True

class FictracTask(Task):
    """
    Task to run fictrac to track the ball movement and save the results in the behaviour dataframe
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "fictrac"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict):
        # print("TODO: implement FictracTask.test_todo() method!!!")
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images"),
                                       name=f"camera_{CURRENT_USER['fictrac_cam']}-*.dat",
                                       raise_error=False) for trial_dir in trial_dirs_todo]
        TODO2 = any([found_file is None for found_file in found_files])
        if not TODO2:
            TODO2 = any([bool(len(found_file)) is None for found_file in found_files])
        return TODO1  or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)
        # this has no inbuilt override protection -
        # -> only protected by the test_todo() method of this Task
        config_and_run_fictrac(fly_dict["dir"], trial_dirs, CURRENT_USER['fictrac_cam'])

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess.get_dfs()
        return True

    def run_multiple_flies(self, fly_dicts, params=None):
        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        if isinstance(self.params, list):
            n_params = len(self.params)
        else:
            n_params = 1
        all_selected_trials = []
        for fly_dict in fly_dicts:
            all_selected_trials += get_selected_trials(fly_dict)

        config_and_run_fictrac("multiple flies", all_selected_trials)

        for i_fly, fly_dict in enumerate(fly_dicts):
            print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
            params = self.params if n_params == 1 else self.params[i_fly]
            trial_dirs = get_selected_trials(fly_dict)
            preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=params,
                                       trial_dirs=trial_dirs)
            preprocess.get_dfs()

class Df3dTask(Task):
    """
    Task to run poase estimation using DeepFly3D and DF3D post processing
    and save results in behaviour dataframe.
    """
    def __init__(self, prio: int=0) -> None:
        super().__init__(prio)
        self.name = "df3d"
        self.previous_tasks = [BehDataTransferTask(), SyncDataTransfer()]

    def test_todo(self, fly_dict: dict) -> bool:
        TODO1 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        trial_dirs_todo = get_selected_trials(fly_dict)
        TODO2 = not all([os.path.isdir(os.path.join(trial_dir, "behData", "images", "df3d"))
                         for trial_dir in trial_dirs_todo])
        if not TODO2:
            found_files = [utils.find_file(os.path.join(trial_dir, "behData", "images", "df3d"),
                                       name="aligned_pose__*.pkl",
                                       raise_error=False) for trial_dir in trial_dirs_todo]
            TODO2 = any([found_file is None for found_file in found_files])
            if not TODO2:
                TODO2 = not all([bool(len(found_file)) is None for found_file in found_files])

        return TODO1 or TODO2

    def run(self, fly_dict: dict, params: PreProcessParams=None) -> bool:
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)
        self.params.overwrite = fly_dict["overwrite"]

        trial_dirs = get_selected_trials(fly_dict)

        print("STARTING PREPROCESSING OF FLY: \n" + fly_dict["dir"])
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        preprocess._pose_estimate()
        preprocess._post_process_pose()

        return True

class VideoTask(Task):
    """
    Task to make videos for each trial that contain one behavioural camera, DFF,
    and registered raw data.
    Additionaly copy the videos to a central folder
    """
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
            TODO = TODO or not os.path.isfile(os.path.join(trial_dir, load.PROCESSED_FOLDER,
                                                           file_name))
            TODO = TODO or not os.path.isfile(os.path.join(CURRENT_USER["video_dir"], file_name))
        return TODO

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.dff_video_max_length = None # 100
        #self.params.dff_video_downsample = 2
        self.params.overwrite = fly_dict["overwrite"]

        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)
        for i, trial_dir in enumerate(trial_dirs):
            preprocess._make_dff_behaviour_video_trial(i_trial=i, mask=None, include_2p=True)
            name_video = f"{preprocess.date}_{preprocess.genotype}_Fly{preprocess.fly}_" +\
                f"{preprocess.trial_names[i]}_{preprocess.params.dff_beh_video_name}.mp4"
            if not os.path.exists(CURRENT_USER["video_dir"]):
                os.makedirs(CURRENT_USER["video_dir"])
            shutil.copy2(
                os.path.join(trial_dir, load.PROCESSED_FOLDER,name_video),
                os.path.join(CURRENT_USER["video_dir"],name_video))

        return True

class LaserStimProcessTask(Task):
    """
    Task to process the synchronisation signals from laser stimulation
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "stim_process"
        self.previous_tasks = [FictracTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name="stim_paradigm.pkl")
        TODO2 = self._test_todo_trials(fly_dict, file_name=global_params.df3d_df_out_dir)
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
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
    """
    Task to make fly through videos and projection plots for averaged 3d volumes
    """
    def __init__(self, prio=0):
        super().__init__(prio)
        self.name = "video_plot_3d"
        self.previous_tasks = [TifTask()]

    def test_todo(self, fly_dict):
        TODO1 = self._test_todo_trials(fly_dict, file_name="zstack.mp4") or \
            self._test_todo_trials(fly_dict, file_name="ystack.mp4") or \
            self._test_todo_trials(fly_dict, file_name="xstack.mp4")
        TODO2 = self._test_todo_trials(fly_dict, file_name="3dproject.png")
        return TODO1 or TODO2

    def run(self, fly_dict, params=None):
        if not self.wait_for_previous_task(fly_dict):
            return False
        else:
            self.send_status_email(fly_dict)
            print(f"{time.ctime(time.time())}: starting {self.name} task for fly {fly_dict['dir']}")

        self.params = deepcopy(params) if params is not None else deepcopy(global_params)

        trial_dirs = get_selected_trials(fly_dict)

        self.params.overwrite = self.params.overwrite = fly_dict["overwrite"]
        preprocess = PreProcessFly(fly_dir=fly_dict["dir"], params=self.params,
                                   trial_dirs=trial_dirs)

        for i, processed_dir in enumerate(preprocess.trial_processed_dirs):
            green_tif = os.path.join(processed_dir, self.params.green_raw)
            green_avg = os.path.join(processed_dir,"green_avg.tif") if os.path.isfile(green_tif) else None

            red_tif = os.path.join(processed_dir, self.params.red_raw)
            red_avg = os.path.join(processed_dir,"red_avg.tif") if os.path.isfile(red_tif) else None
            show3d.make_avg_videos_3d(
                green=green_tif if os.path.isfile(green_tif) else None,
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=green_avg,
                red_avg=red_avg,
                out_dir=processed_dir
            )
            show3d.plot_projections_3d(
                green=green_tif if os.path.isfile(green_tif) else None,
                red=red_tif if os.path.isfile(red_tif) else None,
                green_avg=green_avg,
                red_avg=red_avg,
                out_dir=processed_dir
            )

            shutil.copy2(
                os.path.join(processed_dir, "zstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_zstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "ystack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_ystack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "xstack.mp4"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_xstack.mp4")
            )
            shutil.copy2(
                os.path.join(processed_dir, "3dproject.png"),
                os.path.join(CURRENT_USER["video_dir"],
                             f"{preprocess.date}_{preprocess.genotype}_" +\
                             f"Fly{preprocess.fly}_{preprocess.trial_names[i]}_3dproject.png")
            )

        return True

task_collection = {
    "twop_data_transfer": TwopDataTransferTask(prio=-100),
    "beh_data_transfer": BehDataTransferTask(prio=-100),
    "pre_cluster": PreClusterTask(prio=10),
    "pre_cluster_green_only": PreClusterGreenOnlyTask(prio=8),
    "tif": TifTask(prio=9),
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
    "video_plot_3d": VideoPlot3DTask(prio=-15),
    "replace_moving_frames": ReplaceMovingFrames(prio=10)
}
