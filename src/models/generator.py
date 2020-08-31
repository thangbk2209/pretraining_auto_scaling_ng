from tensorflow.keras import Model, Input
import tensorflow as tf


# class Generator(Model):
#     def __init__(self, encoder, noise_size, mlp_block, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.mlp_block = mlp_block
#         self.noise_size = noise_size
#
#     def call(self, inputs, **kwargs):
#         z = self.encoder(inputs)
#         noise = tf.random.normal(shape=noise_size)
#         z = tf.concat([z, noise], axis=-1)
#         z = self.mlp_block(z)
#         return z


def get_generator(encoder, noise_size,  mlp_block):
    input_data = Input(shape=encoder.input_shape[1:])
    input_noise = Input(shape=(noise_size,))
    z = encoder(input_data)
    z = tf.concat([z, input_noise], axis=-1)
    z = mlp_block(z)
    return Model(inputs=[input_data, input_noise], outputs=[z], name='generator')
