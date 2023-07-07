from standardml.datasets.index import IndexedData, Indexer
from standardml.datasets.index.factory import IndexerFactory, IndexerType
from standardml.datasets.index.indexers import ImgImgIndexerConfig
from standardml.ml.tensorflow.dataparser import TFDatasetParserConfig, TFDatasetParser
from standardml.pipelines.builder import PipelineBuilder

task = 'train'
path_sample = 'tests/resources/datasets/imgimg_black'

pipeline_tasks = 'tests/resources/pipelines/pipeline_image_input_and_label.json'
inputs_pipeline, labels_pipeline = PipelineBuilder.build(pipeline_tasks=pipeline_tasks)

config_default: ImgImgIndexerConfig = ImgImgIndexerConfig(
    path_inputs=path_sample, path_labels=path_sample,
    type='default', rnd_seed=42, task=task,
    val_split=0, test_split=0)

indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
index_data: IndexedData = indexer.index()

dataset_config: TFDatasetParserConfig = TFDatasetParserConfig(
    inputs_shape=[128, 128, 3], labels_shape=[128, 128, 1],
    precision='float16',
)

parser: TFDatasetParser = TFDatasetParser(
    index_data=index_data, config=dataset_config,
    inputs_pipeline=inputs_pipeline, labels_pipeline=labels_pipeline)


def test_dataloader_parser():
    train_dataset, train_size = parser.get_train_dataset(batch_size=1)

    x, y = next(iter(train_dataset))

    assert x.numpy().mean() == 1.0 and y.numpy().mean() == 1.0
    assert train_size == 1
