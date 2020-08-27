from tensorflow.keras import Model
import tensorflow as tf


class Gan(Model):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs, **kwargs):
        z = self.generator(inputs)
        z = tf.reshape(z, [z.shape[0], z.shape[1], 1])
        z = tf.concat([inputs, z], axis=1)
        z = self.discriminator(z)
        return z

