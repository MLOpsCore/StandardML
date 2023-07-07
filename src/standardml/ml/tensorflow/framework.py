from standardml.ml.framework import MLFramework
from standardml.ml.tensorflow.dataparser import TFDatasetParser, TFDatasetParserConfig
from standardml.ml.tensorflow.factory import TFModelFactory
from standardml.ml.tensorflow.mappers import TFLossMapper, TFMetricsMapper, TFOptimizerMapper


class TFFramework(MLFramework):
    """
    Tensorflow framework
    """
    def arch_factory(self, **kwargs) -> TFModelFactory:
        return TFModelFactory(**kwargs)

    def data_parser(self, **kwargs) -> TFDatasetParser:
        return TFDatasetParser(**kwargs)

    def data_parser_config(self, **kwargs) -> TFDatasetParserConfig:
        return TFDatasetParserConfig(**kwargs)

    def loss_mapper(self, *args, **kwargs) -> TFLossMapper:
        return TFLossMapper(**kwargs)

    def metrics_mapper(self, **kwargs) -> TFMetricsMapper:
        return TFMetricsMapper(**kwargs)

    def optimizer_mapper(self, **kwargs) -> TFOptimizerMapper:
        return TFOptimizerMapper(**kwargs)
