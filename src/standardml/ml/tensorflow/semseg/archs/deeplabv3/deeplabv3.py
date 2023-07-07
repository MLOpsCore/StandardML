import tensorflow as tf

from pydantic import validator, PrivateAttr
from tensorflow import keras as k
from tensorflow.python.keras import backend as keras_backend

from standardml.ml.tensorflow.outputs import TFFitOutputs, TFEvalOutputs
from standardml.ml.tensorflow.imodel import TFModel, ModelConfig


def up_sample_block(tensor, size):
    # Bilinear up sampling
    name = tensor.name.split('/')[0] + '_up_sample'
    return k.layers.Lambda(lambda x: tf.image.resize(images=x, size=size),
                           output_shape=size, name=name)(tensor)


def aspp_block(tensor):
    # ASPP: Atrous Spatial Pyramid Pooling
    dims = keras_backend \
        .int_shape(tensor)

    y_pool = k.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = k.layers.Conv2D(filters=256, kernel_size=1, padding='same',
                             kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = k.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = k.layers.Activation('relu', name=f'relu_1')(y_pool)

    y_pool = up_sample_block(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = k.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = k.layers.BatchNormalization(name=f'bn_2')(y_1)
    y_1 = k.layers.Activation('relu', name=f'relu_2')(y_1)

    y_6 = k.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                          kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = k.layers.BatchNormalization(name=f'bn_3')(y_6)
    y_6 = k.layers.Activation('relu', name=f'relu_3')(y_6)

    y_12 = k.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                           kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = k.layers.BatchNormalization(name=f'bn_4')(y_12)
    y_12 = k.layers.Activation('relu', name=f'relu_4')(y_12)

    y_18 = k.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                           kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = k.layers.BatchNormalization(name=f'bn_5')(y_18)
    y_18 = k.layers.Activation('relu', name=f'relu_5')(y_18)

    y = k.layers.concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = k.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                        kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = k.layers.BatchNormalization(name=f'bn_final')(y)
    y = k.layers.Activation('relu', name=f'relu_final')(y)

    return y


MODELS = {
    "ResNet50": k.applications.ResNet50, "ResNet50V2": k.applications.ResNet50V2,
    "ResNet101": k.applications.ResNet101, "ResNet101V2": k.applications.ResNet101V2,
    "ResNet152": k.applications.ResNet152, "ResNet152V2": k.applications.ResNet152V2
}


class DeepLabV3PlusConfig(ModelConfig):
    model: str
    size: int
    n_clases: int
    name: str

    @validator('model')
    def check_model(cls, v):
        assert v in MODELS.keys(), f"{v} not allowed. Only {list(MODELS.keys())} are allowed"
        return v


class DeepLabV3Plus(TFModel):

    _model: k.Model = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: DeepLabV3PlusConfig):
        super().__init__()
        self._model = self.__build_model(config.size, config.model, config.n_clases, config.name)

    @staticmethod
    def __rename_activations_layer(layers):
        i = 1
        for layer in layers:
            if layer.__class__.__name__ == 'Activation':
                layer._name = f"activation_{i}"
                i += 1

    @staticmethod
    def __build_model(size, model, n_classes, name):

        base_model = MODELS[model](
            input_shape=(size, size, 3),
            weights='imagenet',
            include_top=False)

        DeepLabV3Plus.__rename_activations_layer(base_model.layers)

        im_feature_index, x_b_index = 39, 9

        if model == "ResNet101":
            im_feature_index = 90
        elif model == "ResNet152":
            im_feature_index = 141
        elif model == "ResNet50V2":
            im_feature_index, x_b_index = 38, 8
        elif model == "ResNet101V2":
            im_feature_index, x_b_index = 89, 8
        elif model == "ResNet152V2":
            im_feature_index, x_b_index = 140, 8

        image_features = base_model.get_layer(
            f'activation_{im_feature_index}'
        ).output

        x_a = aspp_block(image_features)
        x_a = up_sample_block(tensor=x_a, size=[size // 4, size // 4])

        x_b = base_model.get_layer(f'activation_{x_b_index}').output

        x_b = k.layers.Conv2D(filters=48, kernel_size=1, padding='same',
                              kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
        x_b = k.layers.BatchNormalization(name=f'bn_low_level_projection')(x_b)
        x_b = k.layers.Activation('relu', name='low_level_activation')(x_b)

        x = k.layers.concatenate([x_a, x_b], name='decoder_concat')

        x = k.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                            kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
        x = k.layers.BatchNormalization(name=f'bn_decoder_1')(x)
        x = k.layers.Activation('relu', name='activation_decoder_1')(x)

        x = k.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                            kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
        x = k.layers.BatchNormalization(name=f'bn_decoder_2')(x)
        x = k.layers.Activation('relu', name='activation_decoder_2')(x)

        x = up_sample_block(x, [size, size])

        x = k.layers.Conv2D(n_classes, (1, 1), name='output_layer')(x)
        x = k.layers.Activation("sigmoid")(x)

        model = k.models.Model(inputs=base_model.input, outputs=x, name=name)

        return model

    def fit(self, train_set, valid_set, train_steps, valid_steps,
            epochs, loss, optimizer, metrics) -> TFFitOutputs:

        # callbacks = generate_callbacks(self.working_path, self._model.name)

        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.9997
                layer.epsilon = 1e-5
            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        history = self._model.fit(train_set, epochs=epochs, steps_per_epoch=train_steps,
                                  validation_data=valid_set, validation_steps=valid_steps)

        return TFFitOutputs(model=self._model, history=history)

    def evaluate(self, test_set, test_steps) -> TFEvalOutputs:
        evaluation = self._model.evaluate(test_set, steps=test_steps)
        return TFEvalOutputs(model=self._model, evaluation=evaluation)

    def predict(self, x):
        return self._model.predict(x)
