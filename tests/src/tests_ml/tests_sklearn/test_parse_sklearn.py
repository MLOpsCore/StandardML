import json
import os

from standardml.datasets.index.factory import IndexerFactory
from standardml.datasets.index.indexers.indexer import IndexedData
from standardml.integrations.integrations import IntegrationFactory
from standardml.ml.framework import MLFrameworkFactory

from standardml.ml.sklearn.outputs import SKLearnFitOutputs

from standardml.pipelines.builder import PipelineBuilder
from standardml.runtime.executor import TrainExecutor, FitData
from standardml.runtime.parse import InputConfig


config_dict_1 = json.load(open('tests/resources/input_config_reg_sklearn_train.json'))


def test_parse_1():
    i: InputConfig = InputConfig(**config_dict_1)
    assert i.version == "1.0"


def test_parse_sklearn_and_create_run_structure():
    input_config: InputConfig = InputConfig(**config_dict_1)

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
    train_dataset = parser.get_train_dataset()

    # Create the architecture
    arch_name = input_config.model.arch.name
    arch_config = input_config.model.arch.config
    arch = framework.arch_factory(name=arch_name).create(arch_config)

    # Map the metrics
    metrics = framework.metrics_mapper(
        metrics=input_config.model.procedure.metrics
    ).map()

    # Create the integrations
    component = arch

    if os.environ.get("ENV") == "INT":
        for integration in input_config.metadata.integrations:
            factory = IntegrationFactory(integration=integration.type)
            component = factory.create(component=component, config=integration.config)

    fit_output: SKLearnFitOutputs = TrainExecutor(
        component=component, data=FitData(
            train_set=train_dataset, train_steps=None,
            valid_set=None, valid_steps=None,
            epochs=None,
            optimizer=None, loss=None, metrics=metrics)
    ).run()

    assert fit_output.metrics["mean_squared_error"] > 0
