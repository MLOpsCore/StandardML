import abc
from typing import List, Optional, Tuple, Union, Callable

import numpy as np

from pydantic import BaseModel

from standardml.ml.model import Model
from standardml.ml.sklearn.outputs import SKLearnFitOutputs, SKLearnEvalOutputs


class ModelConfig(BaseModel):
    """
    Base class for _model configurations.
    """
    name: str = 'BaseModel'


class SKLearnModel(Model):
    """
    Scikit-Learn Model Interface, used to standardize ml architectures.
    The framework used is sklearn.
    """

    @classmethod
    def from_params(cls, params, working_directory):
        pass

    @abc.abstractmethod
    def fit(self, train_set, valid_set, train_steps, valid_steps,
            epochs, loss, optimizer, metrics) -> SKLearnFitOutputs:
        """
        Model Training abstract function, calls the training. Only the below
        parameters are used, rest are kept for compatibility

        :param train_set: Tuple of Numpy NDArrays with training features and labels
        :param metrics: List of metrics to be calculated during training time.
        :return: SKLearnFitOutputs object with fitted method and training metrics
        """
        pass

    @abc.abstractmethod
    def evaluate(self, test_set: Tuple[np.ndarray, np.ndarray], test_steps: Optional[int],
                 metrics: List[Union[str, Callable]]) -> SKLearnEvalOutputs:
        """
        Model Evaluation abstract function, calls the evaluation
        :param test_set: Tf.Dataset for testing
        :param test_steps: Not used
        :return: SKLearnEvalOutputs object with the model and evaluation metrics
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
