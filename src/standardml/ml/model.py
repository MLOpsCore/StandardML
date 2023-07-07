import abc

from standardml.base import Component


class Model(Component, metaclass=abc.ABCMeta):
    """
    Interface Model, used for standardize ml architectures.
    """
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> None:
        """
        Model Evaluation abstract function, calls the evaluation
        function of the _model.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_params(cls, *args, **kwargs):
        """
        Loads a _model from a given set of parameters.
        """
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts the output of a given input.
        """
        pass
