import abc
from typing import Tuple, Iterable, Any

from pydantic import BaseModel


class FitOutputs(BaseModel, metaclass=abc.ABCMeta):
    model: Any

    @abc.abstractmethod
    def extract_metrics(self):
        """
        Extracts metrics from the outputs of a
        _model's fit method.
        """
        pass


class EvalOutputs(BaseModel, metaclass=abc.ABCMeta):

    model: Any

    @abc.abstractmethod
    def extract_metrics(self):
        """
        Extracts metrics from the outputs of a
        _model's evaluate method.
        """
        pass


class Component(BaseModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> FitOutputs:
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> EvalOutputs:
        pass


class AbstractFactory(BaseModel, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        pass


class DatasetParserConfig(BaseModel, metaclass=abc.ABCMeta):
    # Dataset precision
    precision: str = 'float16'


class DatasetParser(BaseModel, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_train_dataset(self, batch_size: int) -> Tuple[Iterable, int]:
        pass

    @abc.abstractmethod
    def get_valid_dataset(self, batch_size: int) -> Tuple[Iterable, int]:
        pass

    @abc.abstractmethod
    def get_test_dataset(self, batch_size: int) -> Tuple[Iterable, int]:
        pass


class Mapper(BaseModel, metaclass=abc.ABCMeta):
    """
    Mapper
    """

    @abc.abstractmethod
    def map(self):
        pass


class LossMapper(Mapper, metaclass=abc.ABCMeta):
    """
    Loss mapper
    """
    ...


class MetricsMapper(Mapper, metaclass=abc.ABCMeta):
    """
    Metrics mapper
    """
    ...


class OptimizerMapper(Mapper, metaclass=abc.ABCMeta):
    """
    Optimizer mapper
    """
    ...


class ModelFactory(AbstractFactory):
    name: str

    @abc.abstractmethod
    def create(self, config: dict):
        pass


class MLFramework(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def arch_factory(self, **kwargs) -> ModelFactory:
        pass

    @abc.abstractmethod
    def data_parser(self, **kwargs) -> DatasetParser:
        pass

    @abc.abstractmethod
    def data_parser_config(self, **kwargs) -> DatasetParserConfig:
        pass

    @abc.abstractmethod
    def loss_mapper(self, **kwargs) -> LossMapper:
        pass

    @abc.abstractmethod
    def metrics_mapper(self, **kwargs) -> MetricsMapper:
        pass

    @abc.abstractmethod
    def optimizer_mapper(self, **kwargs) -> OptimizerMapper:
        pass
