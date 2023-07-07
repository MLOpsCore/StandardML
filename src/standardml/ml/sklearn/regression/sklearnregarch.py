import numpy as np
from pydantic import validator, PrivateAttr
from sklearn import linear_model, neighbors, svm, tree, ensemble
from typing import Any, List, Union, Callable

import sklearn.metrics as skmetrics

from standardml.ml.sklearn.outputs import SKLearnFitOutputs, SKLearnEvalOutputs
from standardml.ml.sklearn.imodel import SKLearnModel, ModelConfig


MODELS = {
    "LogisticRegression": linear_model.LogisticRegression,
    "Ridge": linear_model.Ridge,
    "LinearRegression": linear_model.LinearRegression,
    "Neighbors": neighbors.KNeighborsRegressor,
    "SVR": svm.SVR,
    "DecisionTree": tree.DecisionTreeRegressor,
    "RandomForest": ensemble.RandomForestRegressor,
    "GradientBoostingMachine": ensemble.GradientBoostingRegressor,
    "AdaBoostRegressor": ensemble.AdaBoostRegressor
}


class SKLearnConfig(ModelConfig):
    model: str
    params: dict

    @validator('model')
    def check_model(cls, v):
        assert v in MODELS.keys(), f"{v} not allowed. Only {list(MODELS.keys())} are allowed"
        return v


class SKLearnRegressionModel(SKLearnModel):

    _model: Any = PrivateAttr()
    model: Any

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: SKLearnConfig):
        super().__init__()
        self.model = self.__build_model(config.model, config.params)

    @staticmethod
    def __build_model(model, params):

        return MODELS[model](**params)

    def __generate_metrics(self, target: np.ndarray, y_hat: np.ndarray, metrics: list):

        metrics_dict = {}
        for m in metrics:
            if hasattr(m, "__call__"):  # Checking if it is already a function
                metrics_dict[m.__name__] = m(target, y_hat)
            else:  # Assuming a string
                try:
                    m_func = getattr(skmetrics, m)
                    metrics_dict[m] = m_func(target, y_hat)
                except AttributeError:
                    print(f"Metric {m} was not found")
        return metrics_dict

    def fit(self, train_set, valid_set, train_steps, valid_steps,
            epochs, loss, optimizer, metrics) -> SKLearnFitOutputs:
        
        train_X, train_target = train_set

        self.model = self.model.fit(train_X, train_target)

        y_hat = self.model.predict(train_X)

        train_metrics = self.__generate_metrics(train_target, y_hat, metrics)

        return SKLearnFitOutputs(model=self.model, metrics=train_metrics)

    def evaluate(self, test_target: np.ndarray, test_X: np.ndarray,
                 metrics: List[Union[str, Callable]]) -> SKLearnEvalOutputs:
        y_hat = self.model.predict(test_X)
        metrics = self.__generate_metrics(test_target, y_hat, metrics)

        return SKLearnEvalOutputs(model=self.model, evaluation=metrics)

    def predict(self, x):
        return self.model.predict(x)
