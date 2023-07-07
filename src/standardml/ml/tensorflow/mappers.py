import tensorflow as tf
from typing import List, Union

from standardml.base import LossMapper, OptimizerMapper, MetricsMapper


class TFLossMapper(LossMapper):
    name: str
    config: dict

    def map(self):
        return getattr(tf.keras.losses, self.name)(**self.config)


class TFOptimizerMapper(OptimizerMapper):
    name: str
    config: dict

    def map(self):
        return getattr(tf.keras.optimizers, self.name)(**self.config)


class TFMetricsMapper(MetricsMapper):
    metrics: List[Union[str, dict]] = []

    def map(self):
        metrics = []

        for metric in self.metrics:
            if isinstance(metric, str):
                metrics.append(getattr(tf.keras.metrics, metric)(name=metric.lower()))
            else:
                name, config = metric.get('name'), metric.get('config')
                m = getattr(tf.keras.metrics, name)(**config)
                metrics.append(m)

        return metrics
