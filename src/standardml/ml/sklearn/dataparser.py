from typing import Tuple, Any

import numpy as np

from pydantic import BaseModel

from standardml.datasets.index import IndexedData, IndexPack
from standardml.pipelines.base import Pipeline


class SKLearnDatasetParser(BaseModel):
    """
    Dataset parser
    for sklearn framework
    """

    index_data: IndexedData  # Index data
    config: Any = None  # Dataset config - not used
    inputs_pipeline: Pipeline  # Processing pipeline for inputs
    labels_pipeline: Pipeline  # Processing pipeline for labels

    def get_train_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_reg_dataset(self.index_data.train)

    def get_val_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_reg_dataset(self.index_data.valid)

    def get_test_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tensorflow dataset
        :param batch_size: batch size
        :return: tensorflow dataset
        """
        return self._get_reg_dataset(self.index_data.test)

    def _get_reg_dataset(
                self, index_pack: IndexPack
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tensorflow dataset utility function
        :param batch_size: batch size
        :return: tensorflow dataset
        """

        # Scaler needs to be trained on training set maybe
        x = self.inputs_pipeline.execute(index_pack.x['X'])
        y = self.labels_pipeline.execute(index_pack.x['y'])
        return x, y
