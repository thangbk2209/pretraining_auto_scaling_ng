from tensorflow.keras.layers import Layer, Dropout, Dense
from tensorflow.keras.activations import get as get_activation


class MLPBlock(Layer):
    def __init__(self, layer_units, dropout=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = list()
        self.activation = get_activation(activation)
        if dropout:
            for i in range(len(layer_units)):
                self.hidden_layers.append(Dropout(dropout))
                if layer_units[i] == 1:
                    self.hidden_layers.append(Dense(1))
                else:
                    self.hidden_layers.append(Dense(layer_units[i], activation=activation))
        else:
            for i in range(len(layer_units)):
                if layer_units[i] == 1:
                    self.hidden_layers.append(Dense(1))
                else:
                    self.hidden_layers.append(Dense(layer_units[i], activation=activation))

    def call(self, inputs, **kwargs):
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return z
