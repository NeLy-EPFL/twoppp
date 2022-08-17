from twoppp.run.runparams import global_params, CURRENT_USER
from twoppp.run.tasks import task_collection
from twoppp.run.taskmanager import TaskManager

def main() -> None:
    task_manager = TaskManager(task_collection, params=global_params, user_config=CURRENT_USER)
    task_manager.run()

if __name__ == "__main__":
    main()