from standardml.base import ModelFactory
from standardml.ml.tensorflow.imodel import TFModel
from standardml.ml.tensorflow.semseg.archs import *


class ArchitectureType:
    UNET = 'unet'
    DEEPLAB_V3_PLUS = 'deeplab_v3_plus'


MODELS = {
    ArchitectureType.UNET: (Unet, UnetConfig),
    ArchitectureType.DEEPLAB_V3_PLUS: (DeepLabV3Plus, DeepLabV3PlusConfig),
}


class TFModelFactory(ModelFactory):
    name: str

    def create(self, config: dict) -> TFModel:
        """
        Build TF _model from config
        :param config: config
        """
        arch, conf = MODELS.get(self.name, None)
        # Check if architecture is implemented
        assert arch is not None, f"Architecture {arch} not founded!"
        return arch(config=conf(**config))
