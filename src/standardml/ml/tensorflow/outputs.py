from typing import Any

import tensorflow as tf

from standardml.base import FitOutputs


class TFFitOutputs(FitOutputs):
    """
    The outputs of a Tensorflow _model's fit method.
    """
    model: tf.keras.Model
    history: tf.keras.callbacks.History

    class Config:
        arbitrary_types_allowed = True

    def extract_metrics(self):
        return self.history.history


class TFEvalOutputs(FitOutputs):
    """
    The outputs of a Tensorflow _model's evaluate method.
    """
    model: tf.keras.Model
    evaluation: Any

    class Config:
        arbitrary_types_allowed = True

    def extract_metrics(self):
        return self.evaluation # TODO: implement this
