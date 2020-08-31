from tensorflow.keras.layers import Layer, Dropout, Dense
from tensorflow.keras import Input


class MLPBlock(Layer):
    def __init__(self, layer_units, dropout=None, activation=None, last_activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = list()
        self.number_of_layers = len(layer_units)
        for i in range(self.number_of_layers):
            if dropout:
                self.hidden_layers.append(Dropout(dropout))
            self.hidden_layers.append(Dense(
                layer_units[i],
                activation=last_activation if i == self.number_of_layers - 1 else activation
            ))

    def call(self, inputs, **kwargs):
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return z


# def mlp_block(layer_units, dropout=None, activation=None, last_activation=None):

