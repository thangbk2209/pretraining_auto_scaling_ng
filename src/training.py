import tensorflow as tf
from math import ceil
from src.models.autoencoder import LSTMAutoEncoderV1, LSTMAutoEncoderV2
from src.models.mlp import MLPBlock
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.gan import Gan
from src.losses import discriminator_loss, get_generator_loss_function
from config import *

CONFIG = GanConfig()


def build_train_autoencoder(AE_CONFIG, X_train):
    autoencoder = LSTMAutoEncoderV2(
        input_shape=AE_CONFIG['input_shape'],
        layer_units_encoder=AE_CONFIG['layer_units_encoder'],
        layer_units_decoder=AE_CONFIG['layer_units_decoder'],
        timesteps_decoder=AE_CONFIG['timesteps_decoder'],
        drop_out=AE_CONFIG['drop_out'],
        recurrent_drop_out=AE_CONFIG['recurrent_drop_out'],
        activation=AE_CONFIG['activation'],
        recurrent_activation=AE_CONFIG['recurrent_activation']
    )
    autoencoder.model.compile(optimizer=AE_CONFIG['optimizer'], loss=AE_CONFIG['loss'])
    timesteps_decoder = AE_CONFIG['timesteps_decoder']
    autoencoder.model.fit(
        [X_train, X_train[:, -timesteps_decoder - 1:-1, :]], X_train[:, -timesteps_decoder:, :],
        validation_split=AE_CONFIG['validation_split'],
        batch_size=AE_CONFIG['batch_size'],
        shuffle=False,
        verbose=2,
        callbacks=AE_CONFIG['callbacks']
    )
    return autoencoder


# TODO: implements print_status_bar function
def print_status_bar():
    return None


def train_gan(generator, discriminator, X_gen, y_gen, X_dis):
    batch_size = CONFIG.GAN['batch_size']
    num_samples = min(len(X_gen), len(X_dis))
    num_batches = ceil(num_samples / batch_size)
    gen_optimizer = tf.optimizers.get(CONFIG.GAN['gen_optimizer'])
    dis_optimizer = tf.optimizers.get(CONFIG.GAN['dis_optimizer'])

    # get generator total loss function
    w_gan = GanConfig.GAN['w_gan']
    w_reg = GanConfig.GAN['w_reg']
    w_direct = GanConfig.GAN['w_direct']
    threshold = GanConfig.GAN['threshold']
    generator_loss = get_generator_loss_function(w_gan, w_reg, w_direct, threshold)

    for epoch in tf.range(CONFIG.GAN['epochs']):
        for idx_batch in tf.range(num_batches):
            # data
            idx_start = idx_batch*batch_size
            idx_end = idx_start + batch_size
            if idx_end > num_samples:
                idx_end = num_samples
            X_gen_batch = X_gen[idx_start:idx_end]
            y_gen_batch = y_gen[idx_start:idx_end]
            X_dis_batch = X_dis[idx_start:idx_end]

            # feed forward
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                y_gen_pred = generator(X_gen_batch, training=True)
                tmp = tf.reshape(y_gen_pred, [y_gen_pred.shape[0], y_gen_pred.shape[1], 1])
                tmp = tf.concat([X_gen_batch, tmp], axis=1)
                fake_output = discriminator(tmp, training=True)
                real_output = discriminator(X_dis_batch, training=True)

                gen_loss = generator_loss(
                    X_gen_batch, y_gen_batch, y_gen_pred, fake_output,
                )
                dis_loss = discriminator_loss(real_output, fake_output)

            # back prop
            gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
            dis_optimizer.apply_gradients(zip(gradients_of_dis, discriminator.trainable_variables))
            # TODO: add parameters for print_status_bar function
            print_status_bar()
        print_status_bar()


def train():
    # TODO: add data
    # data
    X_gen_train = None
    y_gen_train = None
    X_gen_test = None
    y_gen_test = None
    X_dis_train = None
    X_dis_test = None

    # pre-training autoencoder for generator: gen_ae
    gen_ae = build_train_autoencoder(CONFIG.GEN_AE, X_gen_train)

    # build generator
    gen_mlp_block = MLPBlock(
        CONFIG.GEN_MLP['layer_units'],
        CONFIG.GEN_MLP['drop_out'],
        CONFIG.GEN_MLP['activation'],
        CONFIG.GEN_MLP['last_activation'],
    )
    generator = Generator(gen_ae.encoder, CONFIG.GAN['noise_shape'], gen_mlp_block)

    # pre-training autoencoder for discriminator: dis_ae
    dis_ae = build_train_autoencoder(CONFIG.DIS_AE, X_dis_train)

    # build discriminator
    dis_mlp_block = MLPBlock(
        CONFIG.DIS_MLP['layer_units'],
        CONFIG.DIS_MLP['drop_out'],
        CONFIG.DIS_MLP['activation'],
        CONFIG.DIS_MLP['last_activation'],
    )
    discriminator = Discriminator(dis_ae.encoder, dis_mlp_block)

    # training GAN
    train_gan(generator, discriminator, X_gen_train, y_gen_train, X_dis_train)

    return None

