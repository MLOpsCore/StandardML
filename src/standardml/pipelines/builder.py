import json
from operator import attrgetter
from typing import Union, List, Dict

from pydantic import BaseModel

from standardml.datasets.processing import image, numeric
from standardml.pipelines.base import Pipeline, PipelineTask


class PipelineModules:
    FUNCTION = 'function'
    IMAGE = 'image'
    NUMERIC = 'numeric'

    modules = {
        "function": FUNCTION,
        "image": IMAGE,
        "numeric": NUMERIC
    }

    @staticmethod
    def resolve(module: str, funct: str) -> callable:
        """
        Resolve a function from a module.
        :param module: The name of the module.
        :param funct: The name of the function.
        :return: The function.
        """
        if module == PipelineModules.FUNCTION:
            return eval(funct)
        elif module == PipelineModules.IMAGE:
            return attrgetter(funct)(image)
        elif module == PipelineModules.NUMERIC:
            return attrgetter(funct)(numeric)
        else:
            raise ValueError(f"Unknown module: {module}")


class PipelineBuilder(BaseModel):
    """
    A builder for pipelines from json.
    The json should be a list of dictionaries, each describing a task.
    """

    @staticmethod
    def build_task(pipeline_task_dict: dict) -> PipelineTask:
        """
        Build a pipeline task from a dictionary.
        :param pipeline_task_dict: The dictionary that describes
            the task.
        :return: A pipeline tasks.
        """
        msg = pipeline_task_dict["msg"]
        module = pipeline_task_dict["module"]
        funct = pipeline_task_dict["funct"]
        args = pipeline_task_dict["args"]

        try:
            funct = PipelineModules.resolve(module, funct)
            return PipelineTask(msg=msg, funct=funct, args=args)
        except ValueError as e:
            raise ValueError(f"Error building task: {msg}") from e

    @staticmethod
    def _build_pipeline(pipeline_tasks: list) -> Pipeline:
        """
        Build a pipeline from a dictionary.
        :param pipeline_tasks: The list of tasks that describe the pipeline
            or the path to a json file.
        :return: A pipeline.
        """
        try:
            pipeline = Pipeline()
            for task in pipeline_tasks:
                pipeline.add(PipelineBuilder.build_task(task))
            return pipeline
        except ValueError as e:
            raise ValueError(f"Error building pipeline: {pipeline_tasks}") from e

    @staticmethod
    def build(pipeline_tasks: Union[str, list, dict]
              ) -> Union[Pipeline, List[Pipeline], Dict[str, Pipeline]]:
        """
        Build a pipeline from a dictionary.
        :param pipeline_tasks: The list of tasks that describe the pipeline
            or the path to a json file or a dictionary of pipelines.
        :return: A pipeline.
        """
        if isinstance(pipeline_tasks, str):
            # Load from file if path is given.
            with open(pipeline_tasks) as f:
                pipeline_tasks = json.load(f)

        # If the content is a dictionary we assume it is a dictionary of pipelines.
        if isinstance(pipeline_tasks, dict):
            return {k: PipelineBuilder._build_pipeline(v) for k, v in pipeline_tasks.items()}

        if not isinstance(pipeline_tasks, list):
            raise ValueError("Pipeline tasks should be a list, a dictionary or a path to a json file.")

        if len(pipeline_tasks) == 0:
            raise ValueError("Empty pipeline.")

        if isinstance(pipeline_tasks[0], list):
            return [PipelineBuilder._build_pipeline(x) for x in pipeline_tasks]

        return PipelineBuilder._build_pipeline(pipeline_tasks)
