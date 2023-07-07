from typing import List

from pydantic import PrivateAttr
from tensorflow import keras as k

from standardml.ml.tensorflow.outputs import TFFitOutputs, TFEvalOutputs
from standardml.ml.tensorflow.imodel import TFModel, ModelConfig


class UnetConfig(ModelConfig):
    name: str = 'Unet'
    num_filters: List[int]
    size: int
    in_channels: int = 3


class Unet(TFModel):
    class Config:
        arbitrary_types_allowed = True

    _model: k.Model = PrivateAttr()

    def __init__(self, config: UnetConfig):
        super().__init__()
        self._model = self.__build_model(
            config.num_filters, config.size, config.in_channels, config.name)

    @staticmethod
    def __conv_block(x, filters, amount=2):
        for _ in range(amount):
            x = k.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation('relu')(x)
        return x

    @staticmethod
    def __build_model(num_filters, size, in_channels=3, name='Unet'):

        x = inputs = k.layers.Input((size, size, in_channels))

        skip_x = []

        # Encoder
        for f in num_filters:
            x = Unet.__conv_block(x, f)
            skip_x.append(x)
            x = k.layers.MaxPool2D((2, 2))(x)

        # Bridge
        x = Unet.__conv_block(x, num_filters[-1])

        num_filters.reverse()
        skip_x.reverse()

        # Decoder
        for i, f in enumerate(num_filters):
            x = k.layers.UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = k.layers.Concatenate()([x, xs])
            x = Unet.__conv_block(x, f)

        # Output
        x = k.layers.Conv2D(1, (1, 1), padding="same")(x)
        x = k.layers.Activation("sigmoid")(x)

        return k.Model(inputs, x, name=name)

    def fit(self, train_set, valid_set, train_steps, valid_steps,
            optimizer, loss, metrics, epochs) -> TFFitOutputs:

        # Compile _model
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # Train _model
        history = self._model.fit(train_set, steps_per_epoch=train_steps, epochs=epochs,
                                  validation_data=valid_set, validation_steps=valid_steps)

        return TFFitOutputs(model=self._model, history=history)

        # TODO: Fix this
        # callbacks=callbacks)
        # plot_history(history.history, self.working_path, self._model.name)

    def evaluate(self, test_set, test_steps) -> TFEvalOutputs:
        evaluation = self._model.evaluate(test_set, steps=test_steps)
        return TFEvalOutputs(model=self._model, evaluation=evaluation)

    def predict(self, x):
        return self._model.predict(x)
