from .indexer import Indexer, IndexedData, IndexerConfig


class ImgTabIndexerConfig(IndexerConfig):
    pass


class ImgTabIndexer(Indexer):
    """
    Indexer for images and tabular data.
    There are two ways to index:

    1) Images as input, tabular data as output (text)
    2) Images as input, tabular data as output (csv):
      Needs to specify:
     - The separator
     - The name of columns as labels
    """

    config: ImgTabIndexerConfig

    def index(self) -> IndexedData:
        pass
