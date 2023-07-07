import abc
from typing import Dict, List, Any

from pydantic import BaseModel, validator


class IndexPack(BaseModel):
    """
    IndexPack is a class that represents a pair of
    input and label data.
    """
    x: Dict[str, Any] = {}
    y: Dict[str, Any] = {}


class IndexedData(BaseModel):
    train: IndexPack = IndexPack()
    valid: IndexPack = IndexPack()
    test: IndexPack = IndexPack()


class IndexerConfig(BaseModel):
    """
    Base configuration for indexers
    """
    _task_allowed: List[str] = ['train', 'test', 'all']

    test_split: float = 0.2  # Test split
    val_split: float = 0.2  # Val split

    rnd_seed: int = 42  # Random seed
    task: str = 'train'  # Task to be performed

    @validator('task')
    def task_value_no_allowed(cls, v):
        if v not in cls._task_allowed:
            raise ValueError(
                f'Task {v} not allowed, found:', v,
                'allowed:', cls._task_allowed
            )
        return v


class Indexer(BaseModel, metaclass=abc.ABCMeta):
    """
    Interface ILoader, used for standardize data indexers.
    """
    config: IndexerConfig

    @abc.abstractmethod
    def index(self) -> IndexedData:
        """
        Load the data
        :return: IndexedData: Raw data loaded
        """
        pass



