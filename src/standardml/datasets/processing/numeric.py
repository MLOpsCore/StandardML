from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NumericProcessingLib:

    """
    A library of preprocessing tasks in numerical data.
    """

    @staticmethod
    def apply_minmax_scaling(x):
        scaler = MinMaxScaler()
        return scaler.fit_transform(x)

    @staticmethod
    def apply_standard_scaling(x):
        scaler = StandardScaler()
        
        orig_shape = x.shape
        if x.ndim == 1:
            x = x.reshape((-1, 1))  # Turning into just one feature

        return scaler.fit_transform(x).reshape(orig_shape)
