from tensorflow.keras import Model, Input


# class Discriminator(Model):
#     def __init__(self, encoder, mlp_block, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.mlp_block = mlp_block
#
#     def call(self, inputs, **kwargs):
#         z = self.encoder(inputs)
#         z = self.mlp_block(z)
#         return z


def get_discriminator(encoder, mlp_block):
    inputs = Input(shape=encoder.input_shape[1:])
    z = encoder(inputs)
    z = mlp_block(z)
    return Model(inputs=[inputs], outputs=[z])
