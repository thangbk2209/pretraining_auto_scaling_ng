import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pandas import DataFrame


class Data(object):
    def __init__(self,
                 data,  # numpy array (samples, features)
                 split_ratio=(0.8, 0.2),
                 input_time_steps=10
                 ):
        if data.shape[1] != 1:
            raise Exception('Data object does not support multivariate!')
        self.data = np.array(data.astype('float32'))
        self.data_train, self.data_test = self._split(split_ratio)
        self._transform_data()
        self.X_train, self.y_train = self.create_dataset(input_time_steps, type_dataset='train')
        self.X_test, self.y_test = self.create_dataset(input_time_steps, type_dataset='test')

    def _split(self, ratio):
        n_train = int(len(self.data) * ratio[0])
        data_train, data_test = self.data[:n_train], self.data[n_train:]
        return data_train, data_test

    # @staticmethod
    # def difference(data, interval=1):
    #     diff = list()
    #     for i in range(interval, len(data)):
    #         v = data[i] - data[i - interval]
    #         diff.append(v)
    #     return np.array(diff)

    @staticmethod
    def log_transform(data):
        data = data
        df = DataFrame(data)
        data = df.mask(df <= 0.0).fillna(df.mean(axis=0)).values
        return np.log(data)

    def _transform_data(self):
        # fit training data
        train = self.log_transform(self.data_train)
        # train = self.difference(train, self.interval_diff)

        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(train)
        train = self.standard_scaler.transform(train)

        self.minmax_scaler = MinMaxScaler((-1, 1))
        self.minmax_scaler.fit(train)
        train = self.minmax_scaler.transform(train)
        self.scale_data_train = train

        self.scale_data_test = self.transform(self.data_test)

    def transform(self, data):
        data = self.log_transform(data)
        # data = self.difference(data, self.interval_diff)
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

    def create_dataset(self, input_time_steps, data=None, type_dataset=None):
        """
        data: external data, numpy array (sammples, features)
        type_dataset: 'train', 'test'

        return X_train , y_train
            (samples, timesteps, features), (samples, timesteps)

        """
        if data is None and type_dataset is None:
            raise Exception('Error! data=?? or type_dataset=??')

        if data is None:
            if type_dataset == 'train':
                data = self.scale_data_train
            elif type_dataset == 'test':
                data = self.scale_data_test
            else:
                raise Exception('type_data error! please chose: "train" or "test"')

        Xs, ys = [], []
        for i in range(len(data) - input_time_steps):
            v = data[i:i + input_time_steps]
            Xs.append(v)
            ys.append(data[i+input_time_steps])
        Xs, ys = np.array(Xs), np.array(ys)
        return Xs, ys
