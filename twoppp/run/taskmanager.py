"""
task manager module housing the TaskManager class
"""
from copy import deepcopy
from typing import List, Dict
import time
import numpy as np

from twoppp.pipeline import PreProcessParams

from twoppp.run.runutils import read_fly_dirs, read_running_tasks, check_task_running
from twoppp.run.runutils import send_email, write_running_tasks
from twoppp.run.tasks import Task

class TaskManager():
    """
    class to prioritise and sequentially execute different tasks
    """
    def __init__(self, task_collection: Dict[str, Task],
                 params: PreProcessParams, user_config: dict) -> None:
        """
        class to prioritise and sequentially execute different tasks on different flies and trials.
        Most important component is a list of todos that is managed: self.todo_dicts
        Each todo item will have the following fields:
            - dir: the base directory of the fly, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: whether or not to force an overwrite of the previous results
            - tasks: the name of the task to be done. Later used to index in self.task_collection
            - status: whether the todo is "ready","running", "done", or "waiting"

        Parameters
        ----------
        task_collection : Dict[str, Task]
            a dictionary indexed by strings containing instances of the class Task.
            This dictionary should contain all possible tasks
        params : PreProcessParams
            parameters for pre-processing.
        user_config : dict
            dictionary with user specific parameters, such as file directories, etc.
        """
        self.todo_dicts = []
        # status = "ready", "waiting", "running", "done"
        self.task_collection = task_collection
        self.params = deepcopy(params)
        self.user_config=user_config
        self.txt_file_to_process = self.user_config["txt_file_to_process"]
        self.txt_file_running = self.user_config["txt_file_running"]
        self.clean_exit = False
        self.flies_to_process = []
        self.t_wait_s = 60

        self.check_fly_dirs_to_process()
        for fly_dict in self.flies_to_process:
            for task_name in fly_dict["todos"]:
                todo = deepcopy(fly_dict)
                if task_name.startswith("!"):
                    todo["tasks"] = task_name[1:]
                    todo["overwrite"] = True
                else:
                    todo["tasks"] = task_name
                    todo["overwrite"] = False
                self.add_todo(todo)

    @property
    def status(self) -> List[str]:
        """
        get the current status of each of the todos registered by accessing the status of each dict.
        Either "running", "done", "waiting", "ready"

        Returns
        -------
        List[str]
        """
        return [todo["status"] for todo in self.todo_dicts]

    @property
    def prios(self) -> np.ndarray:
        """
        get the priority levels of each of the todos registered by accesing the prio of the Task
        associated with each todo.

        Returns
        -------
        np.ndarray
        """
        return np.array([self.task_collection[todo["tasks"]].prio for todo in self.todo_dicts])

    @property
    def n_todos(self) -> int:
        """
        get the number of todo items.

        Returns
        -------
        int
        """
        return len(self.todo_dicts)

    def check_fly_dirs_to_process(self) -> None:
        """
        find out for which flies processing has to be run by reading the _fly_dirs_to_process file
        and also the _tasks_running text file.
        Calls the check_which_tasks_todo() method for every fly that is found.
        """
        fly_dicts = read_fly_dirs(self.txt_file_to_process)
        running_tasks = read_running_tasks(self.txt_file_running)
        for fly_dict in fly_dicts:
            fly_dict["todos"] = self.check_which_tasks_todo(fly_dict, running_tasks)
            if len(fly_dict["todos"]) == 0:
                continue
            else:
                self.flies_to_process.append(fly_dict)
        print("Will run processing on the following flies: " +\
              f"\n{[fly['dir'] for fly in self.flies_to_process]}")

    def check_which_tasks_todo(self, fly_dict: dict, running_tasks: List[dict]) -> List[dict]:
        """
        check which tasks to do for a given fly by calling the test_todo() method of each task
        and checking whether a task is already running.

        Parameters
        ----------
        fly_dict : dict
            dictionary with info about the fly to process. Has to contain the following fields:
            - dir: the base directory, where the data is stored
            - tasks: a string describing which tasks todo, e.g.: "pre_cluster,post_cluster"
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
        running_tasks : List[dict]
            a list of tasks that is currently running, e.g., as obtained by read_running_tasks().
            Essentially a list of dictionaries like the "fly_dict" specified above.

        Returns
        -------
        List[dict]
            a list of fly_dicts that have processing yet to be done.
        """
        task_names = fly_dict["tasks"].split(",")
        todos = []
        for task_name in task_names:
            if task_name.startswith("!"):
                task_name_orig = task_name
                task_name = task_name[1:]
                force = True
            else:
                force = False
                task_name_orig = task_name
            if task_name not in self.task_collection.keys():
                print(f"{task_name} is not an available pre-processing step.\n" +
                    f"Available tasks: {self.task_collection.keys()}")
            elif check_task_running(fly_dict, task_name, running_tasks) and \
                self.user_config["check_tasks_running"]:
                print(f"{task_name} is already running for fly {fly_dict['dir']}.")
            else:
                if force or self.task_collection[task_name].test_todo(fly_dict):
                    todos.append(task_name_orig)
        return todos

    def add_todo(self, todo: dict) -> None:
        """add a todo item to the list of todos.
        Will call self.rank_todos() after adding the new item.

        Parameters
        ----------
        todo : dict
            dictionary with info about the fly to process. Has to contain the following fields:
            - dir: the base directory, where the data is stored
            - selected_trials: a string describing which trials to run on,
                               e.g. "001,002" or "all_trials"
            - overwrite: whether or not to force an overwrite of the previous results
            - tasks: the name of the task to be done. Later used to index in self.task_collection
        """
        todo["status"] = "ready"
        self.todo_dicts.append(todo)
        self.rank_todos()

    def remove_todo(self, i_todo: int) -> None:
        """
        remove a todo item from the self.todo_dicts list and from the running_tasks text file.

        Parameters
        ----------
        i_todo : int
            index of the todo in the self.todo_dicts list
        """
        todo = self.todo_dicts.pop(i_todo)
        write_running_tasks(todo, add=False)

    def rank_todos(self) -> None:
        """
        sort todos according to the priority of the Tasks that each of them would be calling
        """
        self.todo_dicts = sorted(self.todo_dicts,
                                 key=lambda todo: -1*self.task_collection[todo["tasks"]].prio)

    def execute_next_task(self) -> bool:
        """
        execute the todo from self.todo_dicts with the highest priority that is not waiting.
        Will check for tasks if they are waiting for a previous tasks. If so, will check for the one
        with the next highest priority.

        Returns
        -------
        bool
            True if a task was run to completion, False if all tasks are "waiting.
        """
        self.rank_todos()
        result = False
        for i_todo, next_todo in enumerate(self.todo_dicts):
            task_name = next_todo["tasks"]
            write_running_tasks(next_todo, add=True)
            self.todo_dicts[i_todo]["status"] = "running"
            result = self.task_collection[task_name].run(fly_dict=next_todo, params=self.params)
            if result:
                self.todo_dicts[i_todo]["status"] = "done"
                self.remove_todo(i_todo)
                return True
            else:
                self.todo_dicts[i_todo]["status"] = "waiting"
                write_running_tasks(next_todo, add=False)
        return False

    def run(self) -> None:
        """
        run the tasks manager to sequentially process all todos from self.todo_dicts.
        """
        try:
            while self.n_todos:
                success = self.execute_next_task()
                if not success:
                    # if all tasks are pending wait before checking from start again
                    time.sleep(self.t_wait_s)
            self.clean_exit = True
        finally:
            print("TASK MANAGER clean up: removing tasks from _tasks_running.txt")
            for todo in self.todo_dicts:
                if todo["status"] == "running":
                    write_running_tasks(todo, add=False)
            subject = "TASK MANAGER clean exit" if self.clean_exit else "TASK MANAGER error exit"
            if self.user_config["send_emails"]:
                send_email(subject, "no msg", receiver_email=self.user_config["email"])
