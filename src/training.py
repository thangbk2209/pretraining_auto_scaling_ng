import shutil
import tensorflow as tf
import numpy as np
from math import ceil
from pandas import read_csv
from src.models.autoencoder import LSTMAutoEncoderV1, LSTMAutoEncoderV2
from src.models.mlp import get_mlp_block
from src.models.generator import get_generator
from src.models.discriminator import get_discriminator
from src.losses import discriminator_loss, get_generator_loss_function
from src.preprocessing import Data
from config import *
from src.utils import plot_result

CONFIG = GanConfig()


def build_train_autoencoder(AE_CONFIG, X_train, X_test):
    print('training Autoencoder ----------------------------------------------------')
    autoencoder = LSTMAutoEncoderV2(
        input_shape=AE_CONFIG['input_shape'],
        layer_units_encoder=AE_CONFIG['layer_units_encoder'],
        layer_units_decoder=AE_CONFIG['layer_units_decoder'],
        timesteps_decoder=AE_CONFIG['timesteps_decoder'],
        dropout=AE_CONFIG['dropout'],
        recurrent_dropout=AE_CONFIG['recurrent_dropout'],
        activation=AE_CONFIG['activation'],
        recurrent_activation=AE_CONFIG['recurrent_activation']
    )
    autoencoder.model.compile(optimizer=AE_CONFIG['optimizer'], loss=AE_CONFIG['loss'])
    autoencoder.model.summary()

    timesteps_decoder = AE_CONFIG['timesteps_decoder']
    autoencoder.model.fit(
        [X_train, X_train[:, -timesteps_decoder - 1:-1, :]], X_train[:, -timesteps_decoder:, :],
        validation_split=AE_CONFIG['validation_split'],
        batch_size=AE_CONFIG['batch_size'],
        epochs=AE_CONFIG['epochs'],
        shuffle=False,
        verbose=2,
        callbacks=AE_CONFIG['callbacks']
    )
    test_err = autoencoder.model.evaluate([X_test, X_test[:, -timesteps_decoder - 1:-1, :]],
                                          X_test[:, -timesteps_decoder:, :],
                                          verbose=2)
    print('autoencoder test root mean square error {}'.format(np.sqrt(test_err)))
    return autoencoder


def train_gan(generator, discriminator, X_gen, y_gen, X_dis, X_gen_test, y_gen_test, noise_size, data_obj=None):
    print('training GAN ----------------------------------------------------')
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

    train_loss_list, test_loss_list = [], []
    n_epochs = CONFIG.GAN['epochs']
    y_gen_test_pred, test_loss = None, None
    rmse_func = lambda y_true, y_pred: np.sqrt(((y_true - y_pred) ** 2).mean())
    for epoch in range(n_epochs):
        for idx_batch in tf.range(num_batches):
            # data
            idx_start = idx_batch * batch_size
            idx_end = idx_start + batch_size
            if idx_end > num_samples:
                idx_end = num_samples
            X_gen_batch = X_gen[idx_start:idx_end]
            y_gen_batch = y_gen[idx_start:idx_end]
            X_dis_batch = X_dis[idx_start:idx_end]

            # feed forward
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                y_gen_pred = generator([X_gen_batch, tf.random.normal(shape=(X_gen_batch.shape[0], noise_size))],
                                       training=True)
                tmp = tf.reshape(y_gen_pred, [y_gen_pred.shape[0], y_gen_pred.shape[1], 1])
                tmp = tf.concat([X_gen_batch, tmp], axis=1)
                fake_output = discriminator(tmp, training=True)
                real_output = discriminator(X_dis_batch, training=True)

                gen_total_loss, _, _, _ = generator_loss(
                    X_gen_batch, y_gen_batch, y_gen_pred, fake_output
                )
                dis_loss = discriminator_loss(real_output, fake_output)

            # back prop
            gradients_of_gen = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
            gradients_of_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
            dis_optimizer.apply_gradients(zip(gradients_of_dis, discriminator.trainable_variables))
        # Evaluate:
        y_gen_train_pred = np.stack(
            [generator(
                [X_gen, tf.random.normal(shape=(X_gen.shape[0], noise_size))],
                training=False
            ).numpy() for _ in range(PREDICTION_TIMES_TRAIN)]
        ).mean(axis=0)
        y_gen_test_pred = np.stack(
            [generator(
                [X_gen_test, tf.random.normal(shape=(X_gen_test.shape[0], noise_size))],
                training=False
            ).numpy() for _ in range(PREDICTION_TIMES_TRAIN)]
        ).mean(axis=0)
        train_loss = rmse_func(y_gen, y_gen_train_pred)
        test_loss = rmse_func(y_gen_test, y_gen_test_pred)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        # print loss:
        print('Epoch {}/{}: rmse: train: {} - test: {} ------'.format(epoch + 1, n_epochs, train_loss, test_loss))

    # Evaluating
    y_gen_test_pred = np.stack(
        [generator(
            [X_gen_test, tf.random.normal(shape=(X_gen_test.shape[0], noise_size))],
            training=False
        ).numpy() for _ in range(PREDICTION_TIMES_EVALUATE)]
    ).mean(axis=0)
    y_gen_test_inv = data_obj.invert_transform(y_gen_test)
    y_gen_test_pred_inv = data_obj.invert_transform(y_gen_test_pred)
    test_loss_inv = rmse_func(y_gen_test_inv, y_gen_test_pred_inv)
    plot_result(train_loss_list, test_loss_list, y_gen_test_inv, y_gen_test_pred_inv, test_loss_inv)
    return train_loss_list, test_loss_list

def train():
    # save config.py
    try:
        shutil.copy(os.path.join(PROJECT_DIR, 'config.py'), os.path.join(CONFIGS_DIR, f'config_{RUN_ID}.py'))
    except Exception as e:
        print(e)

    try:
        os.mkdir(MODELS_DIR)
    except Exception as e:
        print(e)
        return None


    # data
    df = read_csv(DATA_FILE, header=DATA_HEADER)
    data = df.iloc[:, DATA_COLUMN].values.reshape(-1, 1)
    data_obj = Data(data, SPLIT_RATIO, INPUT_TIMESTEPS)

    X_gen_train = data_obj.X_train
    y_gen_train = data_obj.y_train
    X_gen_test = data_obj.X_test
    y_gen_test = data_obj.y_test
    X_dis_train, _ = data_obj.create_dataset(input_timesteps=INPUT_TIMESTEPS + 1, data=data_obj.scale_data_train)
    X_dis_test, _ = data_obj.create_dataset(input_timesteps=INPUT_TIMESTEPS + 1, data=data_obj.scale_data_test)

    # pre-training autoencoder for generator: gen_ae
    gen_ae = build_train_autoencoder(CONFIG.GEN_AE, X_gen_train, X_gen_test)
    gen_encoder = gen_ae.encoder
    tf.keras.models.save_model(gen_encoder, os.path.join(MODELS_DIR, 'gen_encoder.h5'))


    # build generator
    gen_mlp_block = get_mlp_block(
        (gen_ae.encoder.output_shape[1] + CONFIG.GAN['noise_size'],),
        CONFIG.GEN_MLP['layer_units'],
        CONFIG.GEN_MLP['dropout'],
        CONFIG.GEN_MLP['activation'],
        CONFIG.GEN_MLP['last_activation'],
    )
    generator = get_generator(gen_encoder, CONFIG.GAN['noise_size'], gen_mlp_block)
    generator.summary()
    # tf.keras.utils.plot_model(generator, show_shapes=True, to_file='gen_test.png')

    # pre-training autoencoder for discriminator: dis_ae
    dis_ae = build_train_autoencoder(CONFIG.DIS_AE, X_dis_train, X_dis_test)
    dis_encoder = dis_ae.encoder
    tf.keras.models.save_model(dis_encoder, os.path.join(MODELS_DIR, 'dis_encoder.h5'))

    # build discriminator
    dis_mlp_block = get_mlp_block(
        dis_ae.encoder.output_shape[1:],
        CONFIG.DIS_MLP['layer_units'],
        CONFIG.DIS_MLP['dropout'],
        CONFIG.DIS_MLP['activation'],
        CONFIG.DIS_MLP['last_activation'],
    )
    discriminator = get_discriminator(dis_encoder, dis_mlp_block)
    discriminator.summary()

    # training GAN
    train_gan(generator, discriminator, X_gen_train, y_gen_train, X_dis_train, X_gen_test, y_gen_test,
              CONFIG.GAN['noise_size'], data_obj)

    generator.save(os.path.join(MODELS_DIR, 'generator.h5'))
    print('model save to {}'.format(MODELS_DIR))
