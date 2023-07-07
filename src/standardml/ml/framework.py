from standardml.base import AbstractFactory, MLFramework
from standardml.ml.tensorflow.framework import TFFramework
from standardml.ml.sklearn.framework import SKLearnFramework


class MLFrameworkName:
    tensorflow: str = "tensorflow"
    pytorch: str = "pytorch"
    sklearn: str = "sklearn"


class MLFrameworkFactory(AbstractFactory):
    _matrix = {
        MLFrameworkName.tensorflow: TFFramework(),
        MLFrameworkName.sklearn: SKLearnFramework()
    }

    framework: str

    def create(self) -> MLFramework:
        assert self.framework in MLFrameworkFactory._matrix
        return MLFrameworkFactory._matrix[self.framework]
