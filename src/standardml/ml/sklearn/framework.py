from standardml.base import DatasetParserConfig, LossMapper, OptimizerMapper
from standardml.ml.framework import MLFramework
from standardml.ml.sklearn.dataparser import SKLearnDatasetParser
from standardml.ml.sklearn.factory import SKLearnModelFactory
from standardml.ml.sklearn.mappers import SKLearnMetricsMapper


class SKLearnFramework(MLFramework):
    """
    Tensorflow framework
    """
    def arch_factory(self, **kwargs) -> SKLearnModelFactory:
        return SKLearnModelFactory(**kwargs)

    def data_parser(self, **kwargs) -> SKLearnDatasetParser:
        return SKLearnDatasetParser(**kwargs)

    def metrics_mapper(self, **kwargs) -> SKLearnMetricsMapper:
        return SKLearnMetricsMapper(**kwargs)
    
    # Below are not needed in SKLearn - consider removing
    def optimizer_mapper(self, **kwargs) -> OptimizerMapper:
        pass

    def loss_mapper(self, **kwargs) -> LossMapper:
        pass

    def data_parser_config(self, **kwargs) -> DatasetParserConfig:
        pass

