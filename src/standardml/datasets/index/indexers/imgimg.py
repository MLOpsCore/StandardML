from pathlib import Path

from typing import Dict, Tuple, List

from pydantic import validator
from sklearn.model_selection import train_test_split

from .indexer import Indexer, IndexedData, IndexerConfig, IndexPack


class ImgImgIndexerConfig(IndexerConfig):
    _types_allowed: List[str] = ['default', 'voc']

    type: str = 'default'  # Type of loader
    path_inputs: str = None  # Path to input data
    path_labels: str = None  # Path to labels data

    # Ignored if type is 'default'
    train_txt: str = 'train.txt'  # Path to train.txt file
    valid_txt: str = 'val.txt'  # Path to valid.txt file

    @validator('type')
    def type_value_no_allowed(cls, v):
        if v not in cls._types_allowed:
            raise ValueError(
                f'Type {v} not allowed, found:', v,
                'allowed:', cls._types_allowed
            )
        return v


class ImgImgIndexer(Indexer):
    """
    Image-Image loader. Loads dataset with:
    - Images as input
    - Images as output

    The data should be organized in the following ways:

    1) Training and validation data are in the same folder and
    the structure is:

        Structure:
        - Path/to/input: img1.png, img2.png, ...
        - Path/to/labels: lbl1.png, lbl2.png, ...

        The index of images and labels will be
        - img1.png -> lbl1.png
        - img2.png -> lbl2.png
        - ...

    2) Training and validation data are in different folders following
    the VOC structure:

        Structure:
        - Path/to/input:
            - images_folder_1: img1_folder1.png, img2_folder1.png, ...
            - images_folder_2: img1_folder2.png, img2_folder2.png, ...
            - ...
        - Path/to/labels:
            - labels_folder_1: lbl1_folder1.png, lbl2_folder1.png, ...
            - labels_folder_2: lbl1_folder2.png, lbl2_folder2.png, ...
            - ...
        - Path/to/train.txt: (train.txt is a file with the following structure)
            - images_folder_1
            - ...
        - Path/to/val.txt: (val.txt is a file with the following structure)
            - images_folder_2
            - ...

        The index of images and labels will be:
        - img1_folder1.png -> lbl1_folder1.png
        - img2_folder1.png -> lbl2_folder1.png
        - ...

    If some image doesn't have a label, it will be ignored.
    If some label doesn't have an image, it will be ignored.

    """
    config: ImgImgIndexerConfig

    def index(self) -> IndexedData:
        if self.config.type == 'default':
            return self._load_default()

        if self.config.type == 'voc':
            return self._load_voc()

    @staticmethod
    def _load_index_and_path(path: Path, is_default=True, only=()) -> Dict[str, str]:
        """
        Load index and path of images and labels.
        :param path: Path to input or labels data
        :param is_default: If True, the data is in the default structure,
            otherwise it is in the VOC structure.
        :param only: If is_default is False, only the folders in this list
        """

        # Dictionary that will be returned
        to_return = {}

        for f in path.glob("*") if is_default else path.glob("*/*"):

            # Check if is a file
            if not f.is_file():
                continue

            # If is_default is True, we set the index to the file name
            # without extension, otherwise we set the index to the name of
            # the parent directory followed by a forward slash and the file name.
            if is_default:
                to_return[f.stem] = str(f)
            else:
                idx = f"{f.parent.name}/{f.stem}"
                if f.parent.name in only:
                    to_return[idx] = str(f)

        return to_return

    @staticmethod
    def _extract_xy(
            search: list, features: Dict[str, str], labels: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        x, y = {}, {}
        for k in search:
            if k in features and k in labels:
                x[k], y[k] = features.get(k, None), labels.get(k, None)
        return x, y

    def _load_default(self) -> IndexedData:
        # Load the names of images without extension (key) and the full path (value)
        index = self._load_index_and_path(Path(self.config.path_inputs))
        # Load the names of labels without extension (key) and the full path (value)
        labels = self._load_index_and_path(Path(self.config.path_labels))

        # Initialize the splits
        train_x_split, val_x_split, test_x_split = list(index.keys()), [], []

        if self.config.test_split != 0:
            # Apply train-test split
            train_x_split, test_x_split = train_test_split(
                train_x_split, test_size=self.config.test_split, random_state=self.config.rnd_seed)

        if self.config.val_split != 0:
            # Apply train-val split
            train_x_split, val_x_split = train_test_split(
                train_x_split, test_size=self.config.val_split, random_state=self.config.rnd_seed)

        # Initialize the data
        train_x, train_y = self._extract_xy(train_x_split, index, labels)
        val_x, val_y = self._extract_xy(val_x_split, index, labels)
        test_x, test_y = self._extract_xy(test_x_split, index, labels)

        # Return the data
        if self.config.task == 'train':
            return IndexedData(
                train=IndexPack(x=train_x, y=train_y),
                valid=IndexPack(x=val_x, y=val_y)
            )

        if self.config.task == 'test':
            return IndexedData(test=IndexPack(x=test_x, y=test_y))

        return IndexedData(
            train=IndexPack(x=train_x, y=train_y),
            valid=IndexPack(x=val_x, y=val_y),
            test=IndexPack(x=test_x, y=test_y)
        )

    def _load_voc_aux(self, txt_file: str, config: ImgImgIndexerConfig):
        # Load the train and validation set
        with open(txt_file, 'r') as f:
            filter_set = f.read().splitlines()
        # Load the names of images without extension (key) and the full path (value)
        index = self._load_index_and_path(Path(config.path_inputs), is_default=False, only=filter_set)
        # Load the names of labels without extension (key) and the full path (value)
        labels = self._load_index_and_path(Path(config.path_labels), is_default=False, only=filter_set)

        return index, labels

    def _load_voc_train(self) -> IndexedData:
        index, labels = self._load_voc_aux(self.config.train_txt, self.config)
        train_x_split, val_x_split = train_test_split(
            list(index.keys()), test_size=self.config.val_split, random_state=self.config.rnd_seed)

        # Initialize the data
        train_x, train_y = self._extract_xy(train_x_split, index, labels)
        val_x, val_y = self._extract_xy(val_x_split, index, labels)

        return IndexedData(
            train=IndexPack(x=train_x, y=train_y),
            valid=IndexPack(x=val_x, y=val_y)
        )

    def _load_voc_test(self) -> IndexedData:
        index, labels = self._load_voc_aux(self.config.valid_txt, self.config)
        # Initialize the data
        test_x, test_y = self._extract_xy(list(index.keys()), index, labels)

        return IndexedData(test=IndexPack(x=test_x, y=test_y))

    def _load_voc(self) -> IndexedData:

        if self.config.task == 'train':
            return self._load_voc_train()

        if self.config.task == 'test':
            return self._load_voc_test()

        train_index = self._load_voc_train()
        test_index = self._load_voc_test()

        return IndexedData(
            train=train_index.train,
            valid=train_index.valid,
            test=test_index.test
        )
