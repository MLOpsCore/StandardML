import abc
from typing import Union

import tensorflow as tf
from pydantic import BaseModel

from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizer_v1 import Optimizer

from standardml.ml.model import Model
from standardml.ml.tensorflow.outputs import TFFitOutputs


class ModelConfig(BaseModel):
    """
    Base class for _model configurations.
    """
    name: str


class TFModel(Model):
    """
    SemSeg Model Interface, used for standardize ml architectures.
    The framework used is Tensorflow.
    """

    @classmethod
    def from_params(cls, params, working_directory):
        pass

    @abc.abstractmethod
    def fit(self, train_set: tf.data.Dataset, valid_set: tf.data.Dataset, train_steps: int, valid_steps: int,
            epochs: int, loss: Union[Loss, str], optimizer: Union[Optimizer, str],
            metrics: Union[Metric, callable]) -> TFFitOutputs:
        """
        Model Training abstract function, calls the training

        :param train_set: Tf.Dataset for training
        :param valid_set: Tf.Dataset for validation
        :param train_steps: Steps taken to iterate over the whole `train_set`
        :param valid_steps: Steps taken to iterate over the whole `valid_set`
        :param loss: Keras loss function
        :param optimizer: Keras optimizer
        :param metrics: List of metrics to be calculated during training time.
        :param epochs: Amount of epochs for training.
        :return: TensorflowFitOutputs object with fitted method and training metrics
        """
        pass

    @abc.abstractmethod
    def evaluate(self, test_set: tf.data.Dataset, test_steps: int) -> None:
        """
        Model Evaluation abstract function, calls the evaluation
        :param test_set: Tf.Dataset for testing
        :param test_steps: Steps taken to iterate over the whole `test_set`
        :return: None
        """
        pass

    @abc.abstractmethod
    def predict(self, x):
        """
        Predicts the output of a given input.
        :param x:
        :return:
        """
        pass
