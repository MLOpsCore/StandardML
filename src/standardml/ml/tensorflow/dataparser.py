from typing import Tuple, List

import tensorflow as tf

from standardml.base import DatasetParserConfig, DatasetParser
from standardml.datasets.index import IndexedData, IndexPack

from standardml.pipelines.base import Pipeline


class TFDatasetParserConfig(DatasetParserConfig):

    # Dataset shapes
    inputs_shape: List[int]
    labels_shape: List[int]

    @property
    def shapes(self):
        return self.inputs_shape, self.labels_shape


class TFDatasetParser(DatasetParser):
    """
    Semantic segmentation dataset parser
    for tensorflow framework
    """

    index_data: IndexedData  # Index data
    config: TFDatasetParserConfig  # Dataset config
    inputs_pipeline: Pipeline  # Processing pipeline for inputs
    labels_pipeline: Pipeline  # Processing pipeline for labels

    def get_train_dataset(self, batch_size: int) -> Tuple[tf.data.Dataset, int]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_tf_dataset(batch_size, self.index_data.train)

    def get_valid_dataset(self, batch_size: int) -> Tuple[tf.data.Dataset, int]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_tf_dataset(batch_size, self.index_data.valid)

    def get_test_dataset(self, batch_size: int) -> Tuple[tf.data.Dataset, int]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_tf_dataset(batch_size, self.index_data.test)

    def _parse(self, index, index_pack: IndexPack):
        """
        Parse function for tensorflow dataset
        :param index: index of the next pair of images
        :param index_pack: index pack
        :return: parsed feature and label
        """

        def numpy_function(idx):
            # It comes as bytes, so we need to decode it
            idx_decoded = idx.decode()
            input_image = self.inputs_pipeline.execute(
                index_pack.x[idx_decoded])
            label_image = self.labels_pipeline.execute(
                index_pack.y[idx_decoded])
            return input_image, label_image

        x, y = tf.numpy_function(numpy_function, [index], [self.config.precision, self.config.precision])

        x_shape, y_shape = self.config.shapes

        x.set_shape(x_shape)
        y.set_shape(y_shape)

        return x, y

    @staticmethod
    def compute_steps(x, batch):
        steps = len(x) // batch
        return steps + 1 if len(x) % batch != 0 else steps

    def _get_tf_dataset(
            self, batch_size: int, index_pack: IndexPack
    ) -> Tuple[tf.data.Dataset, int]:
        """
        Get tensorflow dataset utility function
        :param batch_size: batch size
        :return: tensorflow dataset
        """

        # Generate tf.dataset from keys from IndexPack.x field.
        # for each key, parse will get the values from IndexPack.x and IndexPack.y.
        dataset = tf.data.Dataset.from_tensor_slices(index_pack.x.keys())
        dataset = dataset.map(
            lambda index: self._parse(index, index_pack),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, self.compute_steps(index_pack.x, batch_size)
