from typing import Any

from standardml.base import FitOutputs


class SKLearnFitOutputs(FitOutputs):
    """
    The outputs of a scikit-learn model's fit method.
    """
    model: Any = None
    metrics: dict

    def extract_metrics(self):
        return self.metrics
    

class SKLearnEvalOutputs(FitOutputs):
    """
    The outputs of a scikit-learn model's evaluate method.
    """
    model: Any
    evaluation: Any

    class Config:
        arbitrary_types_allowed = True

    def extract_metrics(self):
        return self.evaluation  # TODO: implement this
