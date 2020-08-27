from tensorflow.keras import Model
import tensorflow as tf


class Generator(Model):
    def __init__(self, encoder, noise_shape, mlp_block, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.mlp_block = mlp_block
        self.noise_shape = noise_shape

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        noise = tf.random.normal(tf.shape(z))
        z = tf.concat([z, noise], axis=-1)
        z = self.mlp_block(z)
        return z

