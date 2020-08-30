from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM, TimeDistributed,
    Dense, Input
)


class LSTMAutoEncoderV1(object):
    def __init__(self,
                 input_shape,  # (time_steps, features)
                 layer_units_encoder,
                 layer_units_decoder,
                 timesteps_decoder,
                 drop_out=.0,
                 recurrent_drop_out=.0,
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        if layer_units_encoder[-1] != layer_units_decoder[0]:
            raise Exception('number of units in the last encoder layer must equal to number of units in the first '
                            'decoder layer')

        self.input_shape = input_shape
        self.layer_units_encoder = layer_units_encoder
        self.layer_units_decoder = layer_units_decoder
        self.timesteps_decoder = timesteps_decoder
        self.drop_out = drop_out
        self.recurrent_drop_out = recurrent_drop_out
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.encoder, self.model = self._build_models()

    def _build_models(self):
        encoder_num_layers = len(self.layer_units_encoder)
        decoder_num_layers = len(self.layer_units_decoder)

        # encoder
        encoder_input = Input(shape=self.input_shape)
        z = encoder_input
        for i in range(encoder_num_layers):
            z = LSTM(
                units=self.layer_units_encoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
                return_sequences=(i != encoder_num_layers - 1),
                return_state=(i == encoder_num_layers - 1)
            )(z)

        encoder_output, state_h, state_c = z
        encoder_state = [state_h, state_c]
        encoder = Model(inputs=[encoder_input], outputs=[encoder_output])

        # autoencoder
        decoder_input = Input(shape=(self.timesteps_decoder, 1))
        v = LSTM(
            units=self.layer_units_decoder[0],
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            dropout=self.drop_out,
            recurrent_dropout=self.recurrent_drop_out,
            return_sequences=True
        )(decoder_input, initial_state=encoder_state)

        for i in range(1, decoder_num_layers):
            v = LSTM(
                units=self.layer_units_decoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
                return_sequences=True
            )(v)
        v = TimeDistributed(Dense(1))(v)

        autoencoder = Model(inputs=[encoder_input, decoder_input], outputs=[v])
        return encoder, autoencoder


class LSTMAutoEncoderV2(object):
    def __init__(self,
                 input_shape,  # (time_steps, features)
                 layer_units_encoder,
                 layer_units_decoder,
                 timesteps_decoder,
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
        self.encoder, self.model = self._build_models()

    def _build_models(self):
        encoder_num_layers = len(self.layer_units_encoder)
        decoder_num_layers = len(self.layer_units_decoder)

        # encoder
        encoder_input = Input(shape=self.input_shape)
        z = encoder_input
        encoder_output = None
        encoder_states = list()
        for i in range(encoder_num_layers):
            z, state_h, state_c = LSTM(
                units=self.layer_units_encoder[i],
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
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
                dropout=self.drop_out,
                recurrent_dropout=self.recurrent_drop_out,
                return_sequences=True
            )(v, initial_state=encoder_states[i])
        v = TimeDistributed(Dense(1))(v)

        autoencoder = Model(inputs=[encoder_input, decoder_input], outputs=[v])
        return encoder, autoencoder
