"""
Sub-module allowing to run processing in an uncomplicated manner using the TaskManager class.
The workflow is as follows:
1. define your processing parameters and your user config in runparams.py
2. add fly directories that you want to process and the tasks you want to execute to the
   _fly_dirs_to_process.txt file
3. execute the run.py file.

If you want to add additional tasks to the ones defined in tasks.py, you can define your own tasks
by sub-classing from the Task() class and adding the new class to the 'task_collection' in tasks.py.
"""
