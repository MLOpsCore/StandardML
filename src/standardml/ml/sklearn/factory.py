from standardml.base import ModelFactory

from standardml.ml.sklearn.imodel import SKLearnModel
from standardml.ml.sklearn.regression.sklearnregarch import SKLearnRegressionModel, SKLearnConfig


class ArchitectureType:
    REGRESSION = "regression"


MODELS = {
    ArchitectureType.REGRESSION: (SKLearnRegressionModel, SKLearnConfig)
}


class SKLearnModelFactory(ModelFactory):
    name: str

    def create(self, config: dict) -> SKLearnModel:
        """
        Build model from configuration
        :param config:
        """
        arch, conf = MODELS.get(self.name, None)
        # Check if architecture is implemented
        assert arch is not None, f"Architecture {arch} not found!"
        return arch(config=conf(**config))
