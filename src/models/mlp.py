from tensorflow.keras.layers import Layer, Dropout, Dense
from tensorflow.keras import Input, Model


# class MLPBlock(Layer):
#     def __init__(self, layer_units, dropout=None, activation=None, last_activation=None, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_layers = list()
#         self.number_of_layers = len(layer_units)
#         for i in range(self.number_of_layers):
#             if dropout:
#                 self.hidden_layers.append(Dropout(dropout))
#             self.hidden_layers.append(Dense(
#                 layer_units[i],
#                 activation=last_activation if i == self.number_of_layers - 1 else activation
#             ))
#
#     def call(self, inputs, **kwargs):
#         z = inputs
#         for layer in self.hidden_layers:
#             z = layer(z)
#         return z


def get_mlp_block(input_shape, layer_units, dropout=None, activation=None, last_activation=None):
    hidden_layers = list()
    number_of_layers = len(layer_units)
    for i in range(number_of_layers):
        if dropout:
            hidden_layers.append(Dropout(dropout))
        hidden_layers.append(Dense(
            layer_units[i],
            activation=last_activation if i == number_of_layers - 1 else activation
        ))
    inputs = Input(shape=input_shape)
    z = inputs
    for layer in hidden_layers:
        z = layer(z)

    name = 'mlp'
    for unit in layer_units:
        name += str('-{}'.format(unit))
    return Model(inputs=[inputs], outputs=[z], name=name)

