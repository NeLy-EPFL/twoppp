"""
utility functions to run processing using the TaskManager
"""
import os
import smtplib
import ssl
from copy import deepcopy
from typing import List
from pexpect import pxssh

from twoppp import load
from twoppp.run.runparams import CURRENT_USER

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))


def get_selected_trials(fly_dict: dict, twoplinux_trial_names: List[str] = None) -> List[str]:
    """
    reads the "selected_trials" field of the fly_dict and returns a list of trial directories.
    Supplying twoplinux_trial_names allows to add trial directories that have not yet been
    copied to the local machine yet.

    Parameters
    ----------
    fly_dict : dict
        dictionary containing information about the fly.
        Should have at least the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
    twoplinux_trial_names : List[str], optional
        supply the names of trials found on the recording computer. This can help in the case
        where the data has not been completely copied to the local machine yet. by default None

    Returns
    -------
    List[str]
        a list of trial directories

    Raises
    ------
    NameError
        In case fly_dict["selected_trials"] contains an entry that cannot be found.
    """
    trial_dirs = load.get_trials_from_fly(fly_dict["dir"], ignore_error=True)[0]
    if twoplinux_trial_names is not None:
        for trial_name in twoplinux_trial_names:
            if not any([trial_dir.split(os.sep)[-1] == trial_name for trial_dir in trial_dirs]):
                trial_dirs.append(os.path.join(fly_dict["dir"], trial_name))

    if fly_dict["selected_trials"] == "all_trials":
        selected_trials = trial_dirs
    else:
        split_comma = fly_dict["selected_trials"].split(",")
        selected_trials = []
        for trial_start in split_comma:
            matched_trials = [trial_dir for trial_dir in trial_dirs
                              if trial_dir.split(os.sep)[-1].startswith(trial_start)]
            if len(matched_trials) == 1:
                selected_trials += matched_trials
            elif len(matched_trials) == 0:
                raise NameError(f"Could not find trial that starts with {trial_start}" +\
                                f" for fly dir {fly_dict['dir']}.\n"+
                                f"Available trials: {trial_dirs}")
    return selected_trials

def split_fly_dict_trials(fly_dict: dict) -> List[dict]:
    """splits a fly_dict with multiple trials into a list of dictionaries with one trial each

    Parameters
    ----------
    fly_dict : dict
        dictionary containing information about the fly.
        Should have at least the following fields:
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
    Returns
    -------
    List[dict]
        a list of dictionaries like fly_dict,
        but each containing only one trial in the "selected_trials" field
    """
    split_comma = fly_dict["selected_trials"].split(",")
    if len(split_comma) == 1:
        return [fly_dict]
    else:
        fly_dicts_split_trials = []
        fly_dict_copy = deepcopy(fly_dict)
        for trial in split_comma:
            fly_dict_copy = deepcopy(fly_dict)
            fly_dict_copy["selected_trials"] = trial
            fly_dicts_split_trials.append(fly_dict_copy)
        return fly_dicts_split_trials

def get_scratch_fly_dict(fly_dict: dict, scratch_base: str) -> dict:
    """
    convert a local fly dict into one where the "dir" field points to the scratch directory
    of the computing cluster that is mounted to the local computer

    Parameters
    ----------
    fly_dict : dict
        dictionary containing information about the fly.
        Should have at least the following fields:
        - "dir": the base directory of the fly
    scratch_base : str
        base directory of the

    Returns
    -------
    dict
        modified fly_dict
    """
    scratch_fly_dict = deepcopy(fly_dict)
    dir_elements = scratch_fly_dict["dir"].split(os.sep)
    scratch_fly_dir = os.path.join(scratch_base, *dir_elements[-2:])
    scratch_fly_dict["dir"] = scratch_fly_dir
    return scratch_fly_dict

def read_fly_dirs(txt_file: str=os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt")) -> List[dict]:
    """
    reads the supplied text file and returns a list of dictionaries
    with information for each fly to process.
    General requested format of a line in the txt file:
    fly_dir||trial1,trial2||task1,task2,!task3||additional arguments,
    ! before a task forces an overwrite.
    example:
    /mnt/nas2/JB/date_genotype/Fly1||all_trials||pre_cluster,fictrac,post_cluster,denoise,dff,!video

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, by default os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt")

    Returns
    -------
    List[dict]
        fly dict with the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
        - "tasks": a comma separated string containing the names of the tasks todo
        - "args": a list of additional arguments read from the text file.

    """
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    fly_dicts = []
    for line in lines:
        if line.startswith("#") or line == "":
            continue
        strings = line.split("||")
        fly = {
            "dir": strings[0],
            "selected_trials": strings[1],
            "tasks": strings[2],
            "args": strings[3:] if len(strings) > 3 else [],
            "todos": []
        }
        fly_dicts.append(fly)
    return fly_dicts

def read_running_tasks(txt_file: str = os.path.join(LOCAL_DIR, "_tasks_running.txt")) -> List[dict]:
    """
    reads the supplied text file and returns a list of dictionaries
    with information for each task that is running.
    General requested format of a line in the txt file:
    fly_dir||trial1,trial2||task1||additional arguments,
    example:
    /mnt/nas2/JB/date_genotype/Fly1||all_trials||pre_cluster

    Parameters
    ----------
    txt_file : str, optional
        location of the text file, by default os.path.join(LOCAL_DIR, "_tasks_running.txt")

    Returns
    -------
    List[dict]
        fly dict with the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
        - "tasks": a tring containing the name of the task running
        - "args": a list of additional arguments read from the text file.
    """
    with open(txt_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    running_tasks = []
    for line in lines:
        if line.startswith("#") or line == "":
            continue
        strings = line.split("||")
        fly = {
            "dir": strings[0],
            "selected_trials": strings[1],
            "tasks": strings[2],
            "args": strings[3:] if len(strings) > 3 else [],
        }
        running_tasks.append(fly)
    return running_tasks

def write_running_tasks(task: dict, add: bool = True,
                        txt_file: str = os.path.join(LOCAL_DIR, "_tasks_running.txt")) -> None:
    """
    Write or delete from the supplied text file to indicate which tasks are currently running.

    Parameters
    ----------
    task : dict
        dict with the following fields:
        - "dir": the base directory of the fly
        - "selected_trials": a string describing which trials to run on,
                             e.g. "001,002" or "all_trials"
        - "tasks": a comma separated string containing the names of the task running
    add : bool, optional
        if True: add to file. if False: remove from file, by default True
    txt_file : str, optional
        location of the text file, by default os.path.join(LOCAL_DIR, "_tasks_running.txt")
    """
    if add:
        with open(txt_file, "a") as file:
            file.write(f"\n{task['dir']}||{task['selected_trials']}||{task['tasks']}")
    else:
        with open(txt_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        lines_to_write = []
        for line in lines:
            if task["dir"] in line and task["tasks"] in line:
                #TODO: check for correct selected trials
                continue
            if line == "":
                continue
            else:
                lines_to_write.append(line)
        with open(txt_file, "w") as file:
            for line in lines_to_write:
                file.write(line+"\n")

def check_task_running(fly_dict: dict, task_name: str, running_tasks: List[dict]) -> bool:
    """
    search for a particular task on a particular fly in the running_tasks list to check
    whether it is running already.

    Parameters
    ----------
    fly_dict : dict
        dict with the following fields:
        - "dir": the base directory of the fly
    task_name : str
        name of the task
    running_tasks : List[dict]
        list of dictionaries specifying running tasks, each with the following entries
        - "dir": the base directory of the fly
        - "tasks": the name of the task running

    Returns
    -------
    bool
        [description]
    """
    fly_tasks = [this_task for this_task in running_tasks if this_task["dir"] == fly_dict["dir"]]
    correct_tasks = [this_task for this_task in fly_tasks if this_task["tasks"] == task_name]
    # TODO: check whether the task is running on the correct trials
    return bool(len(correct_tasks))


def send_email(subject: str, message: str, receiver_email: str) -> None:
    """
    send an email using the nelydebugging@outlook.com account

    Parameters
    ----------
    subject : str
        subject of the e-mail to be sent
    message : str
        messsage of the e-mail to be sent
    receiver_email : str
        receiver e-mail address.
    """
    smtp_server = "smtp-mail.outlook.com"
    port = 587
    sender_email = "nelydebugging@outlook.com"

    email = "Subject: " + subject + "\n\n" + message

    try:
        with open(os.path.join(LOCAL_DIR, ".pwd")) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        password = lines[0]
    except FileNotFoundError:
        password = input("Input e-mail password. " +\
                         "To avoid entering e-mail password in the future, " +\
                         "safe it in a file called .pwd in the same folder as runutils.py.")

    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context) # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email)
    finally:
        server.quit()

def find_trials_2plinux(fly_dir: str, user_folder: str, twop: bool=False) -> List[str]:
    """
    find trials for a particular fly by ssh-ing into the two-photon linux computer.
    This might be useful in case the data is not copied yet completely and some trials
    only exist on the experiment computer, but not on the local computer

    Parameters
    ----------
    fly_dir : str
        base directory of the fly
    user_folder : str
        name of your user folder on the twop linux machine. usually your initials
    twop : bool, optional
        if True search for 2p data trials. If False, search for behavioural data, by default False

    Returns
    -------
    List[str]
        returns a list of trial names, i.e., the names of the folder of each trial.
    """
    ip_address = CURRENT_USER["2p_linux_ip"]
    user = CURRENT_USER["2p_linux_user"]
    try:
        with open(os.path.join(LOCAL_DIR, ".pwd")) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        password = lines[0]
    except FileNotFoundError:
        password = input("Input 2plinux password. " +\
                         "To avoid entering 2plinux password in the future, " +\
                         "safe it in a file called .pwd in the same folder as runutils.py.")

    twop_base_dir = os.path.join("/mnt/windows_share", user_folder)
    beh_base_dir = os.path.join("/data", user_folder)
    base_dir = twop_base_dir if twop else beh_base_dir
    fly_dir_split = fly_dir.split(os.sep)
    remote_fly_dir = os.path.join(base_dir, *fly_dir_split[-2:])

    try:
        server = pxssh.pxssh()
        server.login(ip_address, user, password)
        server.sendline(f"ls {remote_fly_dir}/*/")
        server.prompt()
        answer = str(server.before)
        """
        example answer is similar to the following: (without the line breaks)
        'ls /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/*/\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/001_xz/:\r\n\x1b[0m\x1b[01;34m2p\x1b
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/002_xz/:\r\n\x1b[01;34m2p\x1b[0m\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/003_cc_vol/:\r\n\x1b[01;34m2p\x1b[0m
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/004_cc_t1_vol/:\r\n\x1b[01;34m2p\x1b'
        """
        if "No such file or directory" in answer:
            server.logout()
            return []
        trial_names = answer.split(remote_fly_dir)[2:]
        trial_names = [trial_name.split(os.sep)[1] for trial_name in trial_names]
        server.logout()
        return trial_names
    except pxssh.ExceptionPxssh as error:
        print("pxssh failed while logging into the 2plinux computer.")
        print(error)
        return []
