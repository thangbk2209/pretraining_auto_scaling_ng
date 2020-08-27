import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pandas import DataFrame


class Data(object):
    def __init__(self,
                 data,  # numpy array (samples, features)
                 split_ratio=(0.7, 0.1, 0.2),
                 interval_diff=1
                 ):
        self.data = np.array(data.astype('float32'))
        self.data_train, self.data_validation, self.data_test = self._split(split_ratio)
        self.interval_diff = interval_diff
        self._transform_data()

    def _split(self, ratio=(0.7, 0.15, 0.15)):
        n_train = int(len(self.data) * ratio[0])
        n_val = int(len(self.data) * ratio[1])
        data_train, data_val, data_test = self.data[:n_train], self.data[n_train:n_train + n_val], self.data[
                                                                                                   n_train + n_val:]
        return data_train, data_val, data_test

    @staticmethod
    def difference(data, interval=1):
        diff = list()
        for i in range(interval, len(data)):
            v = data[i] - data[i - interval]
            diff.append(v)
        return np.array(diff)


    @staticmethod
    def log_transform(data):
        data = data
        df = DataFrame(data)
        data = df.mask(df == 0.0).fillna(df.mean(axis=0)).values
        return np.log(data)

    def _transform_data(self):
        train = self.log_transform(self.data_train)
        train = self.difference(train, self.interval_diff)

        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(train)
        train = self.standard_scaler.transform(train)

        self.minmax_scaler = MinMaxScaler((-1, 1))
        self.minmax_scaler.fit(train)
        train = self.minmax_scaler.transform(train)
        self.scale_data_train = train

        self.scale_data_validation = self.transform(self.data_validation)
        self.scale_data_test = self.transform(self.data_test)

    def transform(self, data):
        data = self.log_transform(data)
        data = self.difference(data, self.interval_diff)
        data = self.standard_scaler.transform(data)
        data = self.minmax_scaler.transform(data)
        return data

    # return data predict, none transformed
    # data_shape: (samples, features)
    def invert_transform(self, data_predict_tranformed):
        data = self.minmax_scaler.inverse_transform(data_predict_tranformed)
        data = self.standard_scaler.inverse_transform(data)
        # invert diff
        # data = data + np.log(data_true_none_transformed)[input_time_steps:-self.interval_diff]
        # invert log
        data = np.exp(data)
        return data

    def create_dataset(
            self, data=None, type_dataset=None,
            input_cols=[0], predict_cols=[0],
            input_time_steps=10, predict_time_steps=1
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

