from typing import Optional, List, Union

from pydantic import BaseModel


class IntegrationInputConfig(BaseModel):
    """
    IntegrationInputConfig is a class that represents the
    integration configuration of the configuration file.

    "integration": {
        "name": "mlflow-tracking",
        "type": "mlflow",
        "config": {}
    }

    """
    type: str
    config: dict


class ProblemInputConfig(BaseModel):
    """
    ProblemInputConfig is a class that represents the problem
    configuration of the configuration file.

    "problem": {
        "name": "semseg",
        "category": "binary",
        "framework": "tensorflow"
    }
    """
    name: str
    category: str
    framework: str


class MetadataInputConfig(BaseModel):
    """
    MetadataInputConfig is a class that represents the metadata
    of the configuration file.

    "metadata": {
        "workspace": "workspace",
        "execution": "local",  # local, container, cloud,
        "integrations": [
            {
                "type": "mlflow",
                "config": {
                    "framework": "tensorflow",
                    "experiment_name": "test",
                    "env": "aws",
                    "tracking_uri": "http://localhost:5000",
                    "endpoint_url": "http://localhost:9000",
                    "access_key": "43kT4tfdbDIlyFZh",
                    "secret_key": "TTv0eZoTttsOSPb81hMz6B3HH7y5Xj2d",
                    "details": {
                        "name": "test",
                        "description": "test description",
                        "tags": {'tag1': 'value1', 'tag2': 'value2'}
                    }
                },
            },
            {
                "type": "webhook",
                "config": {}
            }
        ],
    }
    """
    execution: str
    workspace: str
    integrations: Optional[List[IntegrationInputConfig]]


class DatasetInputConfig(BaseModel):
    """
    TFDatasetParserConfig is a class that represents the dataset
    configuration of the configuration file.

    "dataset": {
        "type": "imgimg",
        "config": {
            "type": "default",
            "path_inputs":  "data_annotated_v1/input",
            "path_labels": "data_annotated_v1/labels",
            "rnd_seed": 42,
            "task": "train",
            "val_split": 0,
            "test_split": 0,
        }
    }

    """
    type: str
    config: dict


class PreProcessingInputConfig(BaseModel):
    """
    PreProcessingInputConfig is a class that represents the preprocessing
    configuration of the configuration file.

    "preprocessing": {
        "inputs": [
            {
                "type": "resize",
                "config": {
                    "size": 128
                }
            }
        ],
        "labels": [
            {
                "type": "resize",
                "config": {
                    "size": 128
                }
            }
        ]
    }
    """
    inputs: List[dict] = []
    labels: List[dict] = []


class PostProcessingInputConfig(BaseModel):
    """
    PostProcessingInputConfig is a class that represents the postprocessing
    configuration of the configuration file.

    "postprocessing": {
        "output": [
            {
                "type": "resize",
                "config": {
                    "size": 128
                }
            }
        ]
    }
    """
    output: List[dict] = []


class ProcessingInputConfig(BaseModel):
    """
    ProcessingInputConfig is a class that represents the processing
    configuration of the configuration file.

    "processing": {
        "pre": {
            "inputs": [
                {
                    "type": "resize",
                    "config": {
                        "size": 128
                    }
                }
            ],
            "labels": [
                {
                    "type": "resize",
                    "config": {
                        "size": 128
                    }
                }
            ]
        },
        "post": {
            "output": [
                {
                    "type": "resize",
                    "config": {
                        "size": 128
                    }
                }
            ]
        }
    }
    """
    pre: PreProcessingInputConfig
    post: PostProcessingInputConfig


class DataInputConfig(BaseModel):
    """
    DataInputConfig is a class that represents the data
    configuration of the configuration file.

    "data": {
        "dataset": {
            "type": "imgimg",
            "config": {
                "type": "default",
                "path_inputs":  "data_annotated_v1",
                "path_labels": "data_annotated_v1",
                "rnd_seed": 42,
                "task": "train",
                "val_split": 0,
                "test_split": 0,
            }
        },
        "processing": {
            "pre": {
                "inputs": []
                "labels": []
            },
            "post": {
                "output": []
            }
        },
        "config": {
            "precision": "float32",
            "im_size": 128,
        }
    }
    """
    dataset: DatasetInputConfig
    processing: ProcessingInputConfig
    config: dict


class ArchInputConfig(BaseModel):
    """
    ModelInputConfig is a class that represents the architecture
    configuration of the configuration file.

    "arch": {
        "name": "Unet",
        "config": {
            "num_filters": [2, 4, 8],
            "name": "Unet"
        }
    }
    """
    name: str
    config: dict


class ProcedureInputConfig(BaseModel):
    """
    ProcedureInputConfig is a class that represents the procedure
    configuration of the configuration file.

    "procedure": {
        "name": "train",
        "params": {
            "epochs": 1,
            "batch_size": 1,
            "optimizer": {
                "name": "Adam",
                "config": {
                    "learning_rate": 0.001
                }
            },
            "loss": {
                "name": "BinaryCrossentropy",
                "config": {}
            },
        },
        "metrics": [
            "BinaryAccuracy",
            "Recall",
            "Precision",
            "MeanIoU"
        ]
    }
    """
    name: str
    params: dict
    metrics: List[Union[str, dict]]


class ModelInputConfig(BaseModel):
    """
    ModelInputConfig is a class that represents the _model
    configuration of the configuration file.

    "_model": {
        "arch": {
            "name": "unet",
            "config": {
                "num_filters": [2, 4, 8],
                "name": "unet",
                "size": 128
                "in_channels": 3,
            }
        },
        "procedure": {
            "name": "train",
            "params": {
            "epochs": 1,
            "batch_size": 1,
            "optimizer": {
                "name": "Adam",
                "config": {
                    "learning_rate": 0.001
                }
            },
            "loss": {
                "name": "BinaryCrossentropy",
                "config": {}
            },
            "metrics": [
                "binary_accuracy",
                "recall",
                "precision"
            ]
        }
    }

    """
    arch: ArchInputConfig
    procedure: ProcedureInputConfig


class InputConfig(BaseModel):
    """
    Config is a class that represents
    the configuration file.
    """
    version: str
    task: str
    metadata: MetadataInputConfig
    problem: ProblemInputConfig
    data: DataInputConfig
    model: ModelInputConfig
