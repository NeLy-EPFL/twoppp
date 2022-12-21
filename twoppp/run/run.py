#!/usr/bin/env python3
"""
run script to run processing with the TaskManager class.
See the README.md file in the same folder for usage instructions
"""
from twoppp.run.runparams import global_params, CURRENT_USER
from twoppp.run.tasks import task_collection
from twoppp.run.taskmanager import TaskManager

def main() -> None:
    """
    main function to initialise and run processing
    """
    task_manager = TaskManager(task_collection, params=global_params, user_config=CURRENT_USER)
    print("Will start task manager with the following tasks:")
    for todo_dict in task_manager.todo_dicts:
        print(f"{todo_dict['tasks']}: {todo_dict['dir']}")
    task_manager.run()

if __name__ == "__main__":
    main()
