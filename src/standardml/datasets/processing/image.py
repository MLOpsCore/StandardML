import cv2
import numpy as np


COLOR_MODES = {
    'rgb': cv2.COLOR_BGR2RGB,
    'gray': cv2.COLOR_BGR2GRAY,
    'hsv': cv2.COLOR_BGR2HSV,
    'hls': cv2.COLOR_BGR2HLS,
    'lab': cv2.COLOR_BGR2LAB,
    'luv': cv2.COLOR_BGR2LUV,
    'xyz': cv2.COLOR_BGR2XYZ,
    'yuv': cv2.COLOR_BGR2YUV,
    'ycrcb': cv2.COLOR_BGR2YCrCb,
}


class ImageProcessingLib:

    """
    A library of Image processing actions.
    """

    @staticmethod
    def apply_read_image(path: str):
        """
        Read an image from a path.
        :param path: The path to the image.
        :return: The image.
        """
        return cv2.imread(path, cv2.IMREAD_COLOR)

    @staticmethod
    def apply_clahe(bgr, clip_limit=2.0, tile_grid_size=(32, 32)):
        """
        Apply CLAHE to an image.
        :param bgr: The image to apply CLAHE to.
        :param clip_limit: The clip limits.
        :param tile_grid_size: The tile grid size.
        :return: The image after CLAHE.
        """

        # Convert the image from the RGB color space to the
        # LAB color space.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        # Split the LAB image into three channels,
        # each containing the intensity of the colors.
        lab_planes = cv2.split(lab)
        # Create a CLAHE object.
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # Apply CLAHE to the L channel.
        lab_planes[0] = clahe.apply(lab_planes[0])
        # Merge the channels.
        lab = cv2.merge(lab_planes)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_gaussian(x, kernel_size):
        """
        Apply Gaussian blur to an image.
        :param x: The image to apply Gaussian blur to.
        :param kernel_size: The kernel size.
        :return: The image after Gaussian blur.
        """
        return cv2.GaussianBlur(x, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_squarer(x, size):
        """
        Convert an image into a square image.
        :param x: The image to convert.
        :param size: The size of the square.
        :return: The square image.
        """
        return cv2.resize(x, (size, size))

    @staticmethod
    def apply_change_mode(x, mode):
        """
        Change the mode of an image.
        :param x: The image to change the mode of.
        :param mode: The mode to change to.
        :return: The image after changing the mode.
        """
        return cv2.cvtColor(x, COLOR_MODES[mode])

    @staticmethod
    def apply_change_dtype(x, dtype):
        """
        Change the value type of image.
        :param x: The image to change the value type of.
        :param dtype: The value type to change to.
        :return: The image after changing the value type.
        """
        return x.astype(dtype)

    @staticmethod
    def apply_channel_normalize(x, mode: str):
        """
        Normalize the channels of an image.
        :param x: The image to normalize the channels of.
        :param mode: The mode of the image.
        :return: The image after normalizing the channels.
        """
        if mode == 'hsv':
            # Normalize the hue channel
            x[:, :, 0] = x[:, :, 0] / 179.
            # Normalize the saturation and value channels
            x[:, :, 1:] = x[:, :, 1:] / 255.
        else:
            # Normalize the channels
            x = x / 255.
        return x

    @staticmethod
    def apply_get_channel(x, channel=None):
        """
        Get a channel from an image.
        :param x: The image to get the channel from.
        :param channel: The channel to get.
        :return: The channel.
        """
        return x if channel is None else x[:, :, channel]

    @staticmethod
    def apply_expand_dims(x, axis: int):
        """
        Expand the dimensions of an image.
        :param x: The image to expand the dimensions of.
        :param axis: The axis to expand the dimensions on.
        :return: The image after expanding the dimensions.
        """
        return x if len(x.shape) != 2 else np.expand_dims(x, axis=axis)

    @staticmethod
    def apply_bitwise_not(x):
        """
        Apply bitwise not to an image.
        :param x: The image to apply bitwise not to.
        :return: The image after applying bitwise not.
        """
        return cv2.bitwise_not(x)

    @staticmethod
    def apply_reshape(x, shape: tuple):
        return np.reshape(x, shape)
