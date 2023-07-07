from standardml.base import AbstractFactory, Component
from standardml.integrations.mlflow import MLFlowIntegration, MLFlowConfig


class IntegrationType:
    MLFLOW = 'mlflow'


class IntegrationFactory(AbstractFactory):

    integration: str

    def create(self, component: Component, config: dict) -> Component:
        if self.integration == IntegrationType.MLFLOW:
            return MLFlowIntegration(component=component, config=MLFlowConfig(**config))
        else:
            raise ValueError('Unknown integration: {}'.format(self.integration))
