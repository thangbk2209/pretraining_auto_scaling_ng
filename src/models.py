from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, InputLayer, RepeatVector,
    TimeDistributed, Dense, Flatten,
    Dropout
)
import os


class LSTMAutoEncoder(object):
    def __init__(self,
                 input_shape=None,  # (time_steps, features)
                 layer_units_encoder=[128, 32],
                 layer_units_decoder=[128, 32],
                 timesteps_decoder=5,
                 drop_out=.0,
                 recurrent_drop_out=.0,
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        self.input_shape = input_shape
        self.layer_units_encoder = layer_units_encoder
        self.layer_units_decoder = layer_units_decoder
        self.timesteps_decoder = timesteps_decoder
        self.drop_out = drop_out
        self.recurrent_drop_out = recurrent_drop_out
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.encoder = self._encoder()
        self.embedding_shape = self.encoder.output_shape[1:]
        self.decoder = self._decoder()
        self.model = self._autoencoder()

    def _encoder(self):
        num_layers = len(self.layer_units_encoder)
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        for i in range(num_layers):
            model.add(LSTM(
                units=self.layer_units_encoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
                return_sequences=(i != num_layers - 1)
            ))

        return model

    def _decoder(self):
        num_layers = len(self.layer_units_decoder)
        model = Sequential()
        model.add(InputLayer(input_shape=self.embedding_shape))
        model.add(RepeatVector(self.timesteps_decoder))
        for i in range(num_layers):
            model.add(LSTM(
                units=self.layer_units_decoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
                return_sequences=True
            ))
        model.add(TimeDistributed(Dense(1)))
        return model

    def _autoencoder(self):
        model = Sequential([
            InputLayer(input_shape=self.input_shape),
            self._encoder(),
            self._decoder()
        ])
        return model

    def summary(self):
        print('--------------- Encoder -----------------: ')
        self.encoder.summary()

        print('/n/n')
        print('--------------- Decoder -----------------: ')
        self.decoder.summary()

        print('/n/n')
        print('--------------- Auto Encoder -----------------: ')
        self.model.summary()

    def plot_model(self, save_dir):
        keras.utils.plot_model(self.encoder, show_shapes=True, to_file=os.path.join(save_dir, 'encoder.png'))
        keras.utils.plot_model(self.decoder, show_shapes=True, to_file=os.path.join(save_dir, 'decoder.png'))
        keras.utils.plot_model(self.model, show_shapes=True, to_file=os.path.join(save_dir, 'auto_encoder.png'))


class MLPNet(object):
    def __init__(self,
                 input_shape,
                 hidden_layer_units=[128, 64, 16],
                 drop_out=.0,
                 hidden_activation='relu'):
        self.input_shape = input_shape
        self.hidden_layer_units = hidden_layer_units
        self.drop_out = drop_out
        self.hidden_activation = hidden_activation
        self.model = self._build_model()

    def _build_model(self):
        num_hidden_layers = len(self.hidden_layer_units)
        has_drop_out = self.drop_out != 0.0

        mlp = Sequential()
        mlp.add(InputLayer(input_shape=self.input_shape))
        if has_drop_out:
            mlp.add(Dropout(self.drop_out))

        for i in range(num_hidden_layers):
            mlp.add(Dense(units=self.hidden_layer_units[i], activation=self.hidden_activation))
            if has_drop_out:
                mlp.add(Dropout(self.drop_out))
        mlp.add(Dense(1))
        return mlp

    def plot_model(self, save_dir):
        keras.utils.plot_model(self.model, show_shapes=True, to_file=os.path.join(save_dir, 'MLP Net.png'))


if __name__ == '__main__':
    import os
    TOP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    save_dir = os.path.join(TOP_DIR, 'plot_models')

    # model = LSTMAutoEncoder(
    #     input_shape=(10, 1),
    #     layer_units_encoder=[128, 32],
    #     layer_units_decoder=[32, 128],
    #     timesteps_decoder=5,
    # )
    #
    # model.summary()
    # model.plot_model(save_dir)

    pred_net = MLPNet(input_shape=32,
                      hidden_layer_units=[128, 64, 16],
                      drop_out=.3,
                      hidden_activation='relu')
    pred_net.plot_model(save_dir)
