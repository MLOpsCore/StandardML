import json
import os

from standardml.datasets.index import IndexedData
from standardml.datasets.index.factory import IndexerFactory
from standardml.integrations import IntegrationFactory
from standardml.ml.framework import MLFrameworkFactory
from standardml.pipelines.builder import PipelineBuilder
from standardml.runtime.executor import Executor, ExecutorTask, TrainExecutor, FitData
from standardml.runtime.parse import InputConfig

config_dict_1 = json.load(open('tests/resources/input_config_semseg_train.json'))


def test_parse_1():
    i: InputConfig = InputConfig(**config_dict_1)
    assert i.version == "1.0"


def test_parse_and_create_run_structure():
    input_config = InputConfig(**config_dict_1)

    # Generate the data indexes
    indexer_type = input_config.data.dataset.type
    indexer_config = input_config.data.dataset.config
    indexer = IndexerFactory(indexer_type=indexer_type).create(indexer_config)
    index_data: IndexedData = indexer.index()

    # Create the pipelines
    inputs_pipeline = PipelineBuilder.build(input_config.data.processing.pre.inputs)
    labels_pipeline = PipelineBuilder.build(input_config.data.processing.pre.labels)

    # Generate the framework
    framework_factory = MLFrameworkFactory(framework=input_config.problem.framework)
    framework = framework_factory.create()

    # Create the parser config
    parser_config = framework.data_parser_config(**input_config.data.config)
    parser = framework.data_parser(index_data=index_data, config=parser_config,
                                   inputs_pipeline=inputs_pipeline,
                                   labels_pipeline=labels_pipeline)

    # Create the dataset generator
    batch_size = input_config.model.procedure.params['batch_size']
    train_dataset, train_size = parser.get_train_dataset(batch_size=batch_size)
    valid_dataset, valid_size = parser.get_valid_dataset(batch_size=batch_size)

    # Create the architecture
    arch_name = input_config.model.arch.name
    arch_config = input_config.model.arch.config
    arch = framework.arch_factory(name=arch_name).create(arch_config)

    epochs = input_config.model.procedure.params['epochs']

    # Map the metrics, optimizer and loss
    metrics = framework.metrics_mapper(
        metrics=input_config.model.procedure.metrics
    ).map()

    optimizer = framework.optimizer_mapper(
        name=input_config.model.procedure.params['optimizer']['name'],
        config=input_config.model.procedure.params['optimizer']['config']
    ).map()

    loss = framework.loss_mapper(
        name=input_config.model.procedure.params['loss']['name'],
        config=input_config.model.procedure.params['loss']['config']
    ).map()

    # Create the integrations
    component = arch

    if os.environ.get("ENV") == "INT":
        for integration in input_config.metadata.integrations:
            factory = IntegrationFactory(integration=integration.type)
            component = factory.create(component=component, config=integration.config)

    TrainExecutor(
        component=component, data=FitData(
            train_set=train_dataset, train_steps=train_size,
            valid_set=valid_dataset, valid_steps=valid_size,
            epochs=epochs,
            optimizer=optimizer, loss=loss, metrics=metrics)
    ).run()
