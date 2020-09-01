from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, InputLayer, RepeatVector,
    TimeDistributed, Dense, Flatten,
    Dropout, Input
)
import os


class LSTMAutoEncoder(object):
    def __init__(self,
                 inputs_shape,  # (timesteps, features)
                 layer_units_encoder,
                 layer_units_decoder,
                 timesteps_decoder,
                 dropout=.0,
                 recurrent_dropout=.0,
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        if layer_units_encoder[-1] != layer_units_decoder[0]:
            raise Exception('number of units in the last encoder layer must equal to number of units in the first '
                            'decoder layer')

        self.inputs_shape = inputs_shape
        self.layer_units_encoder = layer_units_encoder
        self.layer_units_decoder = layer_units_decoder
        self.timesteps_decoder = timesteps_decoder
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.encoder, self.model = self._build_models()

    def _build_models(self):
        encoder_num_layers = len(self.layer_units_encoder)
        decoder_num_layers = len(self.layer_units_decoder)

        # encoder
        encoder_inputs = Input(shape=self.inputs_shape)
        z = encoder_inputs
        for i in range(encoder_num_layers):
            z = LSTM(
                units=self.layer_units_encoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=(i != encoder_num_layers - 1),
                return_state=(i == encoder_num_layers - 1)
            )(z)

        encoder_outputs, state_h, state_c = z
        encoder_state = [state_h, state_c]
        encoder = Model(inputs=[encoder_inputs], outputs=[encoder_outputs])

        # autoencoder
        decoder_inputs = Input(shape=(self.timesteps_decoder, 1))
        v = LSTM(
            units=self.layer_units_decoder[0],
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            return_sequences=True
        )(decoder_inputs, initial_state=encoder_state)

        for i in range(1, decoder_num_layers):
            v = LSTM(
                units=self.layer_units_decoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=True
            )(v)
        v = TimeDistributed(Dense(1))(v)

        autoencoder = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[v])
        return encoder, autoencoder


class LSTMAutoEncoderV2(object):
    def __init__(self,
                 inputs_shape,  # (timesteps, features)
                 layer_units_encoder,
                 layer_units_decoder,
                 timesteps_decoder,
                 dropout=.0,
                 recurrent_dropout=.0,
                 activation='tanh',
                 recurrent_activation='sigmoid'):

        self.inputs_shape = inputs_shape
        self.layer_units_encoder = layer_units_encoder
        self.layer_units_decoder = layer_units_decoder
        self.timesteps_decoder = timesteps_decoder
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.encoder, self.model = self._build_models()

    def _build_models(self):
        encoder_num_layers = len(self.layer_units_encoder)
        decoder_num_layers = len(self.layer_units_decoder)

        # encoder
        encoder_input = Input(shape=self.inputs_shape)
        z = encoder_input
        encoder_output = None
        encoder_states = list()
        for i in range(encoder_num_layers):
            z, state_h, state_c = LSTM(
                units=self.layer_units_encoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=(i != encoder_num_layers - 1),
                return_state=True
            )(z)
            encoder_states.append([state_h, state_c])
        encoder = Model(inputs=[encoder_input], outputs=[z])

        # decoder
        decoder_input = Input(shape=(self.timesteps_decoder, 1))
        v = decoder_input
        for i in range(decoder_num_layers):
            v = LSTM(
                units=self.layer_units_decoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=True
            )(v, initial_state=encoder_states[i])
        v = TimeDistributed(Dense(1))(v)

        autoencoder = Model(inputs=[encoder_input, decoder_input], outputs=[v])
        return encoder, autoencoder


class MLPNet(object):
    def __init__(self,
                 hidden_layer_units,
                 dropout=.0,
                 hidden_activation='relu'):
        self.hidden_layer_units = hidden_layer_units
        self.dropout = dropout
        self.hidden_activation = hidden_activation
        self.model = self._build_model()

    def _build_model(self):
        num_hidden_layers = len(self.hidden_layer_units)
        has_dropout = self.dropout != 0.0
        mlp = Sequential()
        if has_dropout:
            mlp.add(Dropout(self.dropout))
        for i in range(num_hidden_layers):
            mlp.add(Dense(units=self.hidden_layer_units[i], activation=self.hidden_activation))
            if has_dropout:
                mlp.add(Dropout(self.dropout))
        mlp.add(Dense(1))
        return mlp

# if __name__ == '__main__':
#     from tensorflow.keras.utils import plot_model
#     from config import *
#     ae = LSTMAutoEncoder(
#         inputs_shape=(10, 1),
#         layer_units_encoder=[128, 32],
#         layer_units_decoder=[32, 128],
#         timesteps_decoder=5,
#     )
#
#     plot_model(ae.model, to_file= CORE_DATA_DIR+'/plot_models/test.png', show_shapes=True)
