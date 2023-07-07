from standardml.base import AbstractFactory
from standardml.datasets.index import Indexer
from standardml.datasets.index.indexers import *


class IndexerType:
    """
    Indexer types
    """
    IMGIMG = 'imgimg'
    IMGLBL = 'imglbl'
    IMGTAB = 'imgtab'
    TABTAB = 'tabtab'


class IndexerFactory(AbstractFactory):
    """
    Factory for indexers
    """
    _types = {IndexerType.IMGIMG: (ImgImgIndexer, ImgImgIndexerConfig),
              IndexerType.IMGLBL: (ImgLblIndexer, ImgLblIndexerConfig),
              IndexerType.IMGTAB: (ImgTabIndexer, ImgTabIndexerConfig),
              IndexerType.TABTAB: (TabTabIndexer, TabTabIndexerConfig)}

    indexer_type: str

    def create(self, config: dict) -> Indexer:
        """
        Build an indexer
        :param config: Configuration for the indexer
        :return: IndexedData: Indexed data
        """
        if self.indexer_type not in IndexerFactory._types:
            raise ValueError('Unknown indexer type: {}'.format(self.indexer_type))

        # Get the indexer and the configuration
        idx, indexer_config = IndexerFactory._types[self.indexer_type]

        # Return an instance of the indexer
        return idx(config=indexer_config(**config))
