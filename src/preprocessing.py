import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class Data(object):
    def __init__(self,
                 data,  # numpy array (samples, features)
                 split_ratio=(0.7, 0.15, 0.15),
                 scaler=None):  # sklearn scaler, string
        self.data = np.array(data.astype('float32'))
        self.data_train, self.data_validation, self.data_test = self._split(split_ratio)
        self.scaler = self._get_scaler(scaler)
        if self.scaler:
            self.scaler.fit(self.data_train)
            self.scale_data_train = self.scaler.transform(self.data_train)
            self.scale_data_validation = self.scaler.transform(self.data_validation)
            self.scale_data_test = self.scaler.transform(self.data_test)

    def _split(self, ratio=(0.7, 0.15, 0.15)):
        n_train = int(len(self.data) * ratio[0])
        n_val = int(len(self.data) * ratio[1])
        data_train, data_val, data_test = self.data[:n_train], self.data[n_train:n_train + n_val], self.data[
                                                                                                   n_train + n_val:]
        return data_train, data_val, data_test

    @staticmethod
    def _get_scaler(scaler):
        if scaler == 'MinMaxScaler':
            return MinMaxScaler(feature_range=(-1, 1))
        if scaler == 'StandardScaler':
            return StandardScaler()
        if scaler == 'RobustScaler':
            return RobustScaler()
        return None

    def create_dataset(
            self, data=None, type_dataset=None,
            input_cols=[0], predict_cols=[0],
            input_time_steps=1, predict_time_steps=1
    ):
        """
        data: external data, numpy array (sammples, features)
        type_dataset: 'train', 'test', 'validation'

        return X_train , y_train
            (samples, time_steps, features), (samples, time_steps, features)

        """
        if data is None and type_dataset is None:
            print('Error! data=?? or type_dataset=??')
            return None

        if data is None:
            if type_dataset == 'train':
                data = self.scale_data_train
            elif type_dataset == 'validation':
                data = self.scale_data_validation
            elif type_dataset == 'test':
                data = self.scale_data_test
            else:
                print('type_data error!')
                return None

        X = data[:, input_cols]
        y = data[:, predict_cols]

        Xs, ys = [], []
        for i in range(len(X) - input_time_steps - predict_time_steps + 1):
            v = X[i:i + input_time_steps]
            Xs.append(v)
            v = y[i + input_time_steps:i + input_time_steps + predict_time_steps]
            ys.append(v)
        Xs, ys = np.array(Xs), np.array(ys)
        return Xs, ys

    @staticmethod
    def difference_from_first_time_step(X, y):
        X_diff = X - X[:, 0].reshape(X.shape[0], 1, X.shape[2])
        y_diff = y - X[:, 0]
        return X_diff, y_diff

    @staticmethod
    def invert_difference(X, y):
        X_invert = X + X[:, 0].reshape(X.shape[0], 1, X.shape[2])
        y_invert = y + X[:, 0]
        return X_invert, y_invert
