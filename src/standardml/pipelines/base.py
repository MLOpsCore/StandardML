from typing import List, Callable, Any
from pydantic import BaseModel


class PipelineTask(BaseModel):
    """
    A task in a pipeline.
    :param msg: A message to print when the task is executed.
    :param funct: The function to execute.
    :param args: The arguments to pass to the function.
    """
    msg: str
    funct: Callable[[Any, ...], Any]
    args: dict

    def execute(self, value):
        """
        Execute the task.
        :param value: The value to pass to the function.
        :return: The result of the function.
        """
        return self.funct(value, **self.args)


class Pipeline(BaseModel):
    """
    A pipeline of preprocessing tasks.
    """
    tasks: List[PipelineTask] = []

    def add(self, task: PipelineTask, condition: bool = True):
        """
        Add a task to the pipeline.
        :param task: The task to add.
        :param condition: Whether to add the task or not.
        """
        if condition:
            self.tasks.append(task)

    def extend(self, tasks: List[PipelineTask]):
        """
        Add a list of tasks to the pipeline.
        :param tasks: The tasks to add.
        """
        self.tasks.extend(tasks)

    def execute(self, value: Any):
        """
        Execute the pipeline.
        :param value: The value to pass to the first task.
        :return: The result of the last task.
        """
        next_input = value
        for task in self.tasks:
            next_input = task.execute(next_input)
        return next_input

    def __str__(self):
        out = f'Pipeline:\n'
        for i, t in enumerate(self.tasks):
            out += f'{i} - {t.funct.__name__}:({t.args}) - {t.msg}\n'
        return out
