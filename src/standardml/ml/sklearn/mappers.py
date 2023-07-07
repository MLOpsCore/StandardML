from typing import List, Union

import sklearn.metrics as skmetrics

from standardml.base import MetricsMapper


class SKLearnMetricsMapper(MetricsMapper):

    metrics: List[Union[str, dict]] = []
        
    def map(self):
        metrics = []

        for metric in self.metrics:
            if isinstance(metric, str):
                metrics.append(getattr(skmetrics, metric))
            else:
                name, config = metric.get('name'), metric.get('config')
                m = getattr(skmetrics, name)
                metrics.append(m)

        return metrics
