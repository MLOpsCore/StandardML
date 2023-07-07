from standardml.datasets.index import IndexedData, Indexer
from standardml.datasets.index.factory import IndexerFactory, IndexerType
from standardml.datasets.index.indexers import ImgImgIndexerConfig

DATASET_PATH_DEFAULT = 'tests/resources/datasets/imgimg/default'
DATASET_PATH_VOC = 'tests/resources/datasets/imgimg/voc'

DATASET_INPUT = 'input'
DATASET_LABELS = 'labels'

RND_SEED = 42

config_default: ImgImgIndexerConfig = ImgImgIndexerConfig(
    path_inputs=f'{DATASET_PATH_DEFAULT}/{DATASET_INPUT}',
    path_labels=f'{DATASET_PATH_DEFAULT}/{DATASET_LABELS}',
    type='default', rnd_seed=RND_SEED, task='all',
    test_split=0.5, val_split=0.5,
)

config_voc: ImgImgIndexerConfig = ImgImgIndexerConfig(
    path_inputs=f'{DATASET_PATH_VOC}/{DATASET_INPUT}',
    path_labels=f'{DATASET_PATH_VOC}/{DATASET_LABELS}',
    train_txt=f'{DATASET_PATH_VOC}/train.txt', valid_txt=f'{DATASET_PATH_VOC}/test.txt',
    type='voc', rnd_seed=RND_SEED, task='all',
    val_split=0.5, test_split=0.5
)


def test_factory_with_imgimg():
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 1 and len(index_data.train.y) == 1
    assert len(index_data.valid.x) == 1 and len(index_data.valid.y) == 1
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1

    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_voc.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 2 and len(index_data.train.y) == 2
    assert len(index_data.valid.x) == 2 and len(index_data.valid.y) == 2
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1


def test_imgimg_voc_mode_all():
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_voc.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 2 and len(index_data.train.y) == 2
    assert len(index_data.valid.x) == 2 and len(index_data.valid.y) == 2
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1


def test_imgimg_voc_mode_train():
    config_voc.task = 'train'
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_voc.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 2 and len(index_data.train.y) == 2
    assert len(index_data.valid.x) == 2 and len(index_data.valid.y) == 2
    assert len(index_data.test.x) == 0 and len(index_data.test.y) == 0


def test_imgimg_voc_mode_test():
    config_voc.task = 'test'
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_voc.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 0 and len(index_data.train.y) == 0
    assert len(index_data.valid.x) == 0 and len(index_data.valid.y) == 0
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1


def test_imgimg_default_mode_all():
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 1 and len(index_data.train.y) == 1
    assert len(index_data.valid.x) == 1 and len(index_data.valid.y) == 1
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1


def test_imgimg_default_mode_train():
    config_default.task = 'train'
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 1 and len(index_data.train.y) == 1
    assert len(index_data.valid.x) == 1 and len(index_data.valid.y) == 1
    assert len(index_data.test.x) == 0 and len(index_data.test.y) == 0


def test_imgimg_default_mode_test():
    config_default.task = 'test'
    indexer: Indexer = IndexerFactory(indexer_type=IndexerType.IMGIMG).create(config_default.dict())
    index_data: IndexedData = indexer.index()

    assert len(index_data.train.x) == 0 and len(index_data.train.y) == 0
    assert len(index_data.valid.x) == 0 and len(index_data.valid.y) == 0
    assert len(index_data.test.x) == 1 and len(index_data.test.y) == 1
