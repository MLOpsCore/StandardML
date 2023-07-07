from .indexer import Indexer, IndexedData, IndexerConfig


class ImgLblIndexerConfig(IndexerConfig):
    pass


class ImgLblIndexer(Indexer):
    """
    Indexer for images and theirs filename as labels

    1) Images as input, labels are the name of the folder
    """

    config: ImgLblIndexerConfig

    def index(self) -> IndexedData:
        pass
