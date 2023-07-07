import os

import pytest
import tensorflow as tf

from standardml.datasets.index import Indexer
from standardml.datasets.index.factory import IndexerFactory, IndexerType
from standardml.datasets.index.indexers import ImgImgIndexerConfig

from standardml.integrations.mlflow import MLFlowIntegration, MLFlowConfig, MLFlowRunDetails

from standardml.ml.tensorflow.dataparser import TFDatasetParserConfig, TFDatasetParser
from standardml.ml.tensorflow.factory import TFModelFactory, ArchitectureType
from standardml.ml.tensorflow.semseg.archs import UnetConfig

from standardml.pipelines.builder import PipelineBuilder


if os.environ.get('ENV') != 'INT':
    pytest.skip("Skipping integration tests", allow_module_level=True)

    TRACKING_URI = os.environ['TRACKING_URI']
    ENDPOINT_URL = os.environ['ENDPOINT_URL']
    ACCESS_KEY = os.environ['ACCESS_KEY']
    SECRET_KEY = os.environ['SECRET_KEY']

    EXP_NAME = os.environ['EXP_NAME']
    RUN_NAME = os.environ['RUN_NAME']


def test_mlflow_integration():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    task = 'train'
    path_sample = 'tests/resources/datasets/imgimg_black'

    config_default: ImgImgIndexerConfig = ImgImgIndexerConfig(
        path_inputs=path_sample, path_labels=path_sample,
        type='default', rnd_seed=42, task=task,
        val_split=0, test_split=0)

    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
    index_data = indexer.index()

    dataset_config: TFDatasetParserConfig = TFDatasetParserConfig(
        precision='float16', inputs_shape=[128, 128, 3], labels_shape=[128, 128, 1])

    # Create the pipelines
    pipeline_tasks = 'tests/resources/pipelines/pipeline_image_input_and_label.json'
    inputs_pipeline, labels_pipeline = PipelineBuilder.build(
        pipeline_tasks=pipeline_tasks)

    parser: TFDatasetParser = TFDatasetParser(
        index_data=index_data, config=dataset_config,
        inputs_pipeline=inputs_pipeline, labels_pipeline=labels_pipeline)

    train_dataset, train_size = parser.get_train_dataset(batch_size=1)

    arch_config = UnetConfig(name='Unet', num_filters=[4, 8, 16],
                             size=dataset_config.inputs_shape[0], in_channels=3)

    arch = TFModelFactory(name=ArchitectureType.UNET).create(arch_config.dict())
    arch_integrated_mlflow = MLFlowIntegration(
        component=arch,
        config=MLFlowConfig(
            framework='tensorflow',
            experiment_name=EXP_NAME,
            tracking_uri=TRACKING_URI,
            endpoint_url=ENDPOINT_URL,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            details=MLFlowRunDetails(
                name=RUN_NAME,
                description='Description de prueba',
                tags={'tag1': 'value1', 'tag2': 'value2'}
            ),
        )
    )

    arch_integrated_mlflow.fit(
        train_set=train_dataset, valid_set=None,
        train_steps=train_size, valid_steps=0, epochs=3,
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.binary_accuracy,
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
            tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5),
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Accuracy(name='accuracy'),
        ])
