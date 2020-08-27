from tensorflow.keras import Model


class Discriminator(Model):
    def __init__(self, encoder, mlp_block, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.mlp_block = mlp_block

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        z = self.mlp_block(z)
        return z

