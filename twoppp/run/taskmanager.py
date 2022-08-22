import os
from copy import deepcopy
import numpy as np
import time

from typing import List, Any, Optional, Dict

from twoppp.pipeline import PreProcessParams

from twoppp.run.runutils import send_email, write_running_tasks, read_fly_dirs, read_running_tasks, check_task_running

class TaskManager():
    def __init__(self, task_collection: dict, params: PreProcessParams, user_config: dict) -> None:
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
        return [todo["status"] for todo in self.todo_dicts]

    @property
    def prios(self) -> np.ndarray:
        return np.array([self.task_collection[todo["tasks"]].prio for todo in self.todo_dicts])

    @property
    def n_todos(self) -> int:
        return len(self.todo_dicts)

    def check_fly_dirs_to_process(self) -> None:
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
        tasks = fly_dict["tasks"].split(",")
        todos = []
        for task in tasks:
            if task.startswith("!"):
                task_orig = task
                task = task[1:]
                force = True
            else:
                force = False
                task_orig = task
            if task not in self.task_collection.keys():
                print(f"{task} is not an available pre-processing step.\n" +
                    f"Available tasks: {self.task_collection.keys()}")
            elif check_task_running(fly_dict, task, running_tasks) and self.user_config["check_tasks_running"]:
                print(f"{task} is already running for fly {fly_dict['dir']}.")
            else:
                if force or self.task_collection[task].test_todo(fly_dict):
                    todos.append(task_orig)
        return todos

    def add_todo(self, todo: dict) -> None:
        todo["status"] = "ready"
        self.todo_dicts.append(todo)
        self.rank_todos()

    def remove_todo(self, i_todo: int) -> None:
        todo = self.todo_dicts.pop(i_todo)
        write_running_tasks(todo, add=False)

    def rank_todos(self) -> None:
        self.todo_dicts = sorted(self.todo_dicts,
                                 key=lambda todo: -1*self.task_collection[todo["tasks"]].prio)

    def execute_next_task(self) -> None:
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
                N_waiting = np.sum([status == "waiting" for status in self.status])
                if N_waiting > 0 and N_waiting < self.n_todos:
                    break
                    # break in order to check again for the highest priority tasks that were waiting
                    # otherwise continue with task of next priority
                elif N_waiting == self.n_todos:
                    time.sleep(self.t_wait_s)
                    # if all tasks are pending wait before checking from start again
            else:
                self.todo_dicts[i_todo]["status"] = "waiting"
                write_running_tasks(next_todo, add=False)

    def run(self) -> None:
        try:
            while self.n_todos:
                """
                if all([todo["tasks"] == "fictrac" for todo in self.todo_dicts]):
                    self.rank_todos()
                    for i_todo, todo in enumerate(self.todo_dicts):
                        self.todo_dicts[i_todo]["status"] = "running"
                        print(f"TASK MANAGER: starting fictrac task for fly {todo['dir']}")
                    result = self.task_collection["fictrac"].run_multiple_flies(fly_dicts=self.todo_dicts, params=self.params)
                    if result:
                        for i_todo, todo in enumerate(self.todo_dicts):
                            self.todo_dicts[i_todo]["status"] = "done"
                            self.remove_todo(i_todo)
                            #TODO: confirm that de-registration of fictrac tasks works
                else:
                """
                self.execute_next_task()
            self.clean_exit = True
        finally:
            print("TASK MANAGER clean up: removing tasks from _tasks_running.txt")
            for todo in self.todo_dicts:
                if todo["status"] == "running":
                    write_running_tasks(todo, add=False)
            subject = "TASK MANAGER clean exit" if self.clean_exit else "TASK MANAGER error exit"
            if self.user_config["send_emails"]:
                send_email(subject, "no msg", receiver_email=self.user_config["email"])