import abc
from typing import Callable, List, Iterable, Any, Optional, Union, Tuple
import numpy as np

from pydantic import BaseModel

from standardml.base import Component, FitOutputs, LossMapper, OptimizerMapper, MetricsMapper, DatasetParser, MLFramework, \
    EvalOutputs
from standardml.datasets.index import IndexedData
from standardml.datasets.index.factory import IndexerFactory
from standardml.integrations import IntegrationFactory
from standardml.ml.framework import MLFrameworkFactory
from standardml.ml.model import Model
from standardml.pipelines.builder import PipelineBuilder
from standardml.runtime.parse import InputConfig


class ExecutorTask:
    TRAIN = 'train'
    TEST = 'test'
    ALL = 'all'


class FitData(BaseModel):
    train_set: Iterable
    train_steps: Optional[int]
    valid_set: Optional[Iterable]
    valid_steps: Optional[int]
    epochs: Optional[int]

    optimizer: Any
    loss: Any
    metrics: Any


class EvalData(BaseModel):
    test_set: Iterable
    test_steps: Optional[int]


class Executor(BaseModel, metaclass=abc.ABCMeta):
    component: Component

    @abc.abstractmethod
    def run(self):
        pass


class TrainExecutor(Executor):
    data: FitData  # TODO: check whether to generalize this (as dict? Any?)

    def run(self) -> FitOutputs:
        return self.component.fit(**self.data.dict())


class EvalExecutor(Executor):
    data: EvalData

    def run(self) -> EvalOutputs:
        return self.component.evaluate(**self.data.dict())


class TrainTestExecutor(Executor):
    data: FitData
    test_data: EvalData

    def run(self) -> Tuple[FitOutputs, EvalOutputs]:
        fit_outputs = self.component.fit(**self.data.dict())
        eval_outputs = self.component.evaluate(**self.test_data.dict())
        return fit_outputs, eval_outputs


class InputConfigExtractor:

    def __init__(self, input_config: InputConfig):
        self.input_config = input_config

    def extract(self) -> Executor:

        framework_factory = MLFrameworkFactory(framework=self.input_config.problem.framework)
        framework = framework_factory.create()

        batch_size = self.input_config.model.procedure.params['batch_size']
        dataset_parser = self.extract_dataset_parser(framework=framework)
        metrics, loss, optimizer = self.extract_metrics_loss_optimizer(framework=framework)

        if self.input_config.task == ExecutorTask.TRAIN:
            train_set, train_steps = dataset_parser.get_train_dataset(batch_size=batch_size)
            valid_set, valid_steps = dataset_parser.get_valid_dataset(batch_size=batch_size)

            component = self.apply_integrations(component=self.extract_arch(framework=framework))

            return TrainExecutor(component=component,
                                 data=FitData(
                                     train_set=train_set, train_steps=train_steps,
                                     valid_set=valid_set, valid_steps=valid_steps,
                                     epochs=self.input_config.model.procedure.params['epochs'],
                                     optimizer=optimizer, loss=loss, metrics=metrics)
                                 )

        if self.input_config.task == ExecutorTask.TEST:
            # TODO: Implement test task
            return EvalExecutor(compenent=None, data=None)

        if self.input_config.task == ExecutorTask.ALL:
            train_set, train_steps = dataset_parser.get_train_dataset(batch_size=batch_size)
            valid_set, valid_steps = dataset_parser.get_valid_dataset(batch_size=batch_size)
            test_set, test_steps = dataset_parser.get_test_dataset(batch_size=batch_size)

            component = self.apply_integrations(component=self.extract_arch(framework=framework))

            return TrainTestExecutor(component=component,
                                     data=FitData(
                                         train_set=train_set, train_steps=train_steps,
                                         valid_set=valid_set, valid_steps=valid_steps,
                                         epochs=self.input_config.model.procedure.params['epochs'],
                                         optimizer=optimizer, loss=loss, metrics=metrics),
                                     test_data=EvalData(test_set=test_set, test_steps=test_steps))

    def apply_integrations(self, component: Component) -> Component:
        for integration in self.input_config.metadata.integrations:
            factory = IntegrationFactory(integration=integration.type)
            component = factory.create(component=component, config=integration.config)
        return component

    def extract_arch(self, framework: MLFramework) -> Model:
        arch_name = self.input_config.model.arch.name
        arch_config = self.input_config.model.arch.config
        return framework.arch_factory(name=arch_name).create(arch_config)

    def extract_dataset_parser(self, framework: MLFramework) -> DatasetParser:
        # Generate the data indexes
        indexer_type = self.input_config.data.dataset.type
        indexer_config = self.input_config.data.dataset.config
        indexer = IndexerFactory(indexer_type=indexer_type).create(indexer_config)
        index_data: IndexedData = indexer.index()

        # Create the pipelines
        inputs_pipeline = PipelineBuilder.build(self.input_config.data.processing.pre.inputs)
        labels_pipeline = PipelineBuilder.build(self.input_config.data.processing.pre.labels)

        # Generate the parser config
        parser_config = framework.data_parser_config(**self.input_config.data.config)
        return framework.data_parser(index_data=index_data, config=parser_config,
                                     inputs_pipeline=inputs_pipeline,
                                     labels_pipeline=labels_pipeline)

    def extract_metrics_loss_optimizer(
            self, framework: MLFramework
    ) -> Tuple[MetricsMapper, LossMapper, OptimizerMapper]:
        # Map the metrics, optimizer and loss
        metrics = framework.metrics_mapper(
            metrics=self.input_config.model.procedure.metrics
        ).map()

        optimizer = framework.optimizer_mapper(
            name=self.input_config.model.procedure.params['optimizer']['name'],
            config=self.input_config.model.procedure.params['optimizer']['config']
        ).map()

        loss = framework.loss_mapper(
            name=self.input_config.model.procedure.params['loss'],
            config=self.input_config.model.procedure.params['loss_config']
        ).map()

        return metrics, loss, optimizer
