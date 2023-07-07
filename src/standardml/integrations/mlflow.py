import os
from typing import Optional

import mlflow
from pydantic import BaseModel

from standardml.base import Component, EvalOutputs
from standardml.ml.tensorflow.outputs import FitOutputs


class MLFlowRunDetails(BaseModel):
    name: str
    description: str
    tags: dict


class MLFlowConfig(BaseModel):
    # The MLFlow framework to use.
    env: str = 'aws'
    framework: str
    experiment_name: str

    details: Optional[MLFlowRunDetails] = None

    # The MLFlow tracking URI.
    tracking_uri: str  # 'MLFLOW_TRACKING_URI'
    endpoint_url: str  # 'MLFLOW_S3_ENDPOINT_URL'
    access_key: str  # 'AWS_ACCESS_KEY_ID'
    secret_key: str  # 'AWS_SECRET_ACCESS_KEY'

    def apply_env_vars(self):
        os.environ['MLFLOW_TRACKING_URI'] = self.tracking_uri
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key


class MLFlowIntegration(Component):
    # The component to be integrated with MLFlow.
    component: Component
    config: MLFlowConfig

    def __init__(self, component: Component, config: MLFlowConfig):
        super().__init__(component=component, config=config)

        # Apply the environment variables.
        self.config.apply_env_vars()

        # Get the MLFLow framework to work with.
        framework_module = getattr(mlflow, self.config.framework)
        framework_module.autolog(
            log_input_examples=True, log_models=True
        )

        mlflow.set_experiment(self.config.experiment_name)

    def evaluate(self, *args, **kwargs) -> EvalOutputs:
        ...

    def fit(self, *args, **kwargs) -> FitOutputs:
        with mlflow.start_run(
                run_name=self.config.details.name,
                description=self.config.details.description,
                tags=self.config.details.tags
        ):
            # Run the component's train method.
            output: FitOutputs = self.component.fit(*args, **kwargs)

        return output
