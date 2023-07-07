from typing import List

import numpy as np

from pydantic import validator
from sklearn.model_selection import train_test_split

from .indexer import Indexer, IndexedData, IndexerConfig, IndexPack


class TabTabIndexerConfig(IndexerConfig):
    _types_allowed: List[str] = ['npz', 'single_csv', 'feat_target_csv', 'folder']

    type: str = 'feat_target_csv'  # Type of loader
    path_input: str = None  # Path to input data
    path_labels: str = None  # Path to labels data

    name_feat: str = 'X' # Name of the input variable in npz
    name_labels: str = 'y' # Name of the target variable in npz

    target_column: str = None # Target column for single csv

    # Ignored if type is 'default'
    train_txt: str = 'train.txt'  # Path to train.txt file
    valid_txt: str = 'val.txt'  # Path to valid.txt file

    @validator('type')
    def type_value_no_allowed(cls, v):
        if v not in cls._types_allowed:
            raise ValueError(
                f'Type {v} not allowed, found:', v,
                'allowed:', cls._types_allowed
            )
        return v


class TabTabIndexer(Indexer):
    """
    TabTabIndexer is a class that represents an
    indexer for tabular data.

    1) Read all from 1 csv file or parquet file.
    Needs to specify:
     - The separator
     - The name of columns as input
     - The name of columns as labels

    2) Read from multiple csv files or parquet files.
    Needs to specify:
     - The separator
     - The name of columns as input
     - The name of columns as labels

    3) Input is a folder with csv files or parquet files.
    Needs to specify:
     - The separator of inputs files
     - The name of columns as input
     - The separator of outputs files
     - The name of columns as labels
    """
    config: TabTabIndexerConfig

    def index(self) -> IndexedData:
        if self.config.type == "npz":
            return self._load_npz()
        else:
            raise NotImplementedError()

    def _load_npz(self) -> IndexedData:
        data = np.load(self.config.path_input)

        X = data[self.config.name_feat]
        y = data[self.config.name_labels]

        # TODO: Train test split may be general, consider moving to parent class
        # Initialize the splits
        train_x, train_y = X, y
        val_x, val_y, test_x, test_y = [], [], [], []

        if self.config.test_split != 0:
            # Apply train-test split
            train_x, test_x, train_y, test_y = train_test_split(
                train_x, train_y, test_size=self.config.test, random_state=self.config.rnd_seed)

        if self.config.val_split != 0:
            # Apply train-val split
            train_x, val_x, train_y, val_y = train_test_split(
                train_x, train_y, test_size=self.config.val, random_state=self.config.rnd_seed)

        return IndexedData(
            train=IndexPack(x={'X': train_x, 'y': train_y}),
            val=IndexPack(x={'X': val_x, 'y': val_y}),
            test=IndexPack(x={'X': test_x, 'y': test_y}),
        )
