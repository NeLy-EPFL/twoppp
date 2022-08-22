import os
import smtplib, ssl
from copy import deepcopy
from pexpect import pxssh
from typing import List

from twoppp import load

LOCAL_DIR, _ = os.path.split(os.path.realpath(__file__))


def get_selected_trials(fly_dict: dict, twoplinux_trial_names: List[str]=None) -> List[str]:
    trial_dirs = load.get_trials_from_fly(fly_dict["dir"], ignore_error=True)[0]
    if twoplinux_trial_names is not None:
        for trial_name in twoplinux_trial_names:
            if not any([trial_dir.split(os.sep)[-1] == trial_name for trial_dir in trial_dirs]):
                trial_dirs.append(os.path.join(fly_dict["dir"], trial_name))

    if fly_dict["selected_trials"] == "all_trials":
        selected_trials = trial_dirs
    else:
        # split_dash = fly_dict["selected_trials"].split("-")
        split_comma = fly_dict["selected_trials"].split(",")
        # if len(split_dash) == 2 and all([split_d.isnumeric() for split_d in split_dash]):
        #     trials_dirs_todo = trial_dirs[split_dash[0]:split_dash[1]+1]
        # elif all([split_c.isnumeric() for split_c in split_comma]):
        #     selected_trials = [trial_dirs[int(i_t)] for i_t in split_comma]
        selected_trials = []
        for trial_start in split_comma:
            matched_trials = [trial_dir for trial_dir in trial_dirs if trial_dir.split(os.sep)[-1].startswith(trial_start)]
            if len(matched_trials) == 1:
                selected_trials += matched_trials
            elif len(matched_trials) == 0:
                raise NameError(f"Could not find trial that starts with {trial_start} for fly dir {fly_dict['dir']}.\n"+
                                f"Available trials: {trial_dirs}")
        # else:
        #     raise NotImplementedError(f"Format string for selected trials of fly {fly_dict['dir']} could not be read."+
        #         "Format string should be of the type of one of the following: 'all_trials', '1-3', or '1,3,4,5'.")
    return selected_trials

def get_scratch_fly_dict(fly_dict, scratch_base):
    scratch_fly_dict = deepcopy(fly_dict)
    dir_elements = scratch_fly_dict["dir"].split(os.sep)
    scratch_fly_dir = os.path.join(scratch_base, *dir_elements[-2:])
    scratch_fly_dict["dir"] = scratch_fly_dir
    return scratch_fly_dict

def read_fly_dirs(txt_file=os.path.join(LOCAL_DIR, "_fly_dirs_to_process.txt")):
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

def read_running_tasks(txt_file=os.path.join(LOCAL_DIR, "_tasks_running.txt")):
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

def write_running_tasks(task, add=True, txt_file=os.path.join(LOCAL_DIR, "_tasks_running.txt")):
    # TODO: check whether task running
    # running_tasks = read_running_tasks(txt_file)
    # is_running = check_task_running(task, task["tasks"], running_tasks):
    # if not is_running
    if add:
        with open(txt_file, "a") as file:
            file.write(f"\n{task['dir']}||{task['selected_trials']}||{task['tasks']}")
    else:
        with open(txt_file) as file:
            lines = file.readlines()
        lines_to_write = []
        for line in lines:
            if task["dir"] in line and task["tasks"] in line:
                #TODO: check for correct selected trials
                continue
            else:
                lines_to_write.append(line)
        with open(txt_file, "w") as file:
            for line in lines_to_write:
                file.write(line)

def check_task_running(fly_dict, task, running_tasks):
    fly_tasks = [this_task for this_task in running_tasks if this_task["dir"] == fly_dict["dir"]]
    correct_tasks = [this_task for this_task in fly_tasks if this_task["tasks"] == task]
    # TODO: check whether the task is running on the correct trials
    return bool(len(correct_tasks))


def send_email(subject, message, receiver_email):
    """send an email using the nelydebugging@outlook.com account

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
        password = input("Input e-mail password. To avoid entering e-mail password in the future, " +\
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
    IP = "128.178.198.12"
    user = "dalco"
    try:
        with open(os.path.join(LOCAL_DIR, ".pwd")) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        password = lines[0]
    except FileNotFoundError:
        password = input("Input 2plinux password. To avoid entering 2plinux password in the future, " +\
                         "safe it in a file called .pwd in the same folder as runutils.py.")
    
    twop_base_dir = os.path.join("/mnt/windows_share", user_folder)
    beh_base_dir = os.path.join("/data", user_folder)

    fly_dir_split = fly_dir.split(os.sep)
    remote_fly_dir = os.path.join(twop_base_dir, *fly_dir_split[-2:]) if twop else os.path.join(beh_base_dir, *fly_dir_split[-2:])

    try:
        s = pxssh.pxssh()
        s.login(IP, user, password)
        s.sendline(f"ls {remote_fly_dir}/*/")
        s.prompt()
        answer = str(s.before)

        """
        example answer: (without the line breaks)
        'ls /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/*/\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/001_xz/:\r\n\x1b[0m\x1b[01;34m2p\x1b[0m\r\n\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/002_xz/:\r\n\x1b[01;34m2p\x1b[0m\r\n\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/003_cc_vol/:\r\n\x1b[01;34m2p\x1b[0m\r\n\r\n
        /mnt/windows_share/JB/220721_DfdxGCaMP6s_tdTom/Fly2/004_cc_t1_vol/:\r\n\x1b[01;34m2p\x1b[0m\r\n'
        """
        if "No such file or directory" in answer:
            s.logout()
            return []
        trial_names = answer.split(remote_fly_dir)[2:]
        trial_names = [trial_name.split(os.sep)[1] for trial_name in trial_names]
        s.logout()
        return trial_names
    except pxssh.ExceptionPxssh as e:
        print("pxssh failed while logging into the 2plinux computer.")
        print(e)
        return []