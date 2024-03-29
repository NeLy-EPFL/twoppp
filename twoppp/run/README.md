# New interface to automatically run multiple processing steps
these instructions are specific to the infrastructure available at the Neuroengineering Laboratory at EPFL

## General procedure:
1. specify fly directories and the tasks you want to perform in [twoppp/run/_fly_dirs_to_process.txt](_fly_dirs_to_process.txt) according to the examples specified therein
    - if this file cannot be found, try: [twoppp/run/_fly_dirs_to_process_example.txt](_fly_dirs_to_process_example.txt)
    - all available tasks and their names can be found in [twoppp/run/tasks.py](tasks.py)
2. add your user configuration as a dictionary and set parameters in [twoppp/run/runparams.py](runparams.py). Set `DEFAULT_USER` to your user, or put `export TWOPPP_USER=YOURINITIALS` in your `.bashrc`
3. if you want to implement new tasks or variations of tasks, subclass from Task() in [tasks.py](tasks.py) and add them to the task_collection list in [tasks.py](tasks.py)
4. run the [twoppp/run/run.py](run.py) script: ```./run.py```

## Additional Requirements:
1. If you want to run fictrac on your data, install it [according to instructions](https://github.com/rjdmoore/fictrac):
    - install it in the folder '~/bin', so that the fictrac executable ends up in '~/bin/fictrac/bin/fictrac'
2. Mount the the server where your data is located, to your workstation, for example at /mnt/nas2. (Refer to the lab manual for details.)
3. Mount your scratch server from the cluster to your workstation, for example as follows for the user jbraun:
    - ```sudo sshfs -o allow_other jbraun@jed.epfl.ch:/scratch/jbraun /mnt/scratch/jbraun```
    - make sure to either permanently mount the scratch or to run the command again when you reboot your workstation
4. save the user password for the twop_linux machine in a file called .pwd
    - not required if you don't want to check for trials on the twop linux machine and don't want to send status e-mails
        - set "check_2plinux_trials" and "send_emails" to False in your user parameters in [runparams.py](runparams.py)
5. If you want to automatically run HandBrake on all videos created, install it [according to the instractions in the docstring of the handbrake function](../plot/videos.py). HandBrake reduces the file size of the video and makes sure they can be easily shown in PowerPoint or Keynote. This is optional and will not throw errors in case it is not installed.

## running motion correction on the cluster:
Also check out the [documentation pages of scitas](https://scitas-data.epfl.ch/kb)
1. ask Kate to get access to FIDIS
2. log in with your gaspar user account
3. create a virtual environment with python 3.10 and call it ofco
    1. ```module load gcc/11.3 python/3.10.4```
    2. ```mkdir venvs```
    3. ```python -m venv $HOME/venvs/ofco```
    4. ```source $HOME/venvs/ofco/bin/activate```
4. install [utils2p](https://github.com/NeLy-EPFL/utils2p) and [ofco](https://github.com/NeLy-EPFL/ofco) into the venv
5. create a folder called registration (```mkdir $HOME/registration```) and copy the following files from the twoppp package to the cluster:
    - [warping_cluster.py](../register/warping_cluster.py): this is the python script that will be run on an individual trial
    - [run_all_folders.sh](../register/run_all_folders.sh): This shell script browses through your scratch directory and finds trial directories for which the processing has not been completely run. For each of these trials, one job will be started on the cluster.
        -  --> update your username in the file (i.e., replace jbraun by your username)
        - adjust the time that you want to request the cluster for. You will only pay for what you actually use, but your job starts faster when you only request shorter jobs. trials of 10000 frames (\~10 mins) usually finish in less than 40h and trials of 4100 frames (\~4minutes) usually finish in less than 16h
    - [run_one_folder.sh](../register/run_one_folder.sh): This shell script only submits one job to the cluster for one specific folder
6. Run the "pre_cluster" task (see [tasks.py](tasks.py)) to perform center of mass registration and copy the necessary data to your scratch directory 
7. Activate the virtualenv (```source $HOME/venvs/ofco/bin/activate```) before submitting cluster jobs. (Consider putting that command in your fidis `.bashrc` so it always gets run.) Then submit cluster jobs in one of the two ways:
    - ```./registration/run_all_folders.sh``` --> careful: this will also re-start jobs that are already running, because they are not yet completed.
    - ```./registration/run_one_folder.sh /scratch/$USER/DATE_GENOTPYE/FLYX/00X_TRIAL_DIR/processed```
8. Checking the status of your jobs: ```squeue -u $USER```
9. Cancelling jobs if something was not correct: ```scancel JOB_ID``` or ```scancel -u $USER```
10. If you have the "cluster" task added to your tasks in [_fly_dirs_to_process.txt](_fly_dirs_to_process.txt) (see [tasks.py](tasks.py)), then this will regularly print how far along the cluster registration is.
11. Run the "post_cluster" task (see [tasks.py](tasks.py)) to copy the results back from the cluster and finalise the motion correction.
