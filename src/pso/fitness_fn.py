from tensorflow.keras.layers import LSTM
from tensorflow.keras import Model, Input
import tensorflow as tf
from pandas import read_csv
from math import ceil
import numpy as np
import multiprocessing

from src.models.mlp import get_mlp_block
from src.preprocessing import Data
from src.losses import get_generator_loss_function, discriminator_loss, mean_squared_func
from config import *

gan_config = GanConfig()


def build_model(result_config):
    # generator
    input_data = Input(shape=(result_config['gen_input_timesteps'], 1))  # (timesteps, features)
    input_noise = Input(shape=(result_config['gen_noise_shape']))
    z = LSTM(
        units=2 ** result_config['gen_lstm_units'],
        dropout=result_config['gen_lstm_dropout'],
        recurrent_dropout=result_config['gen_lstm_recurrent_dropout'],
        return_sequences=False
    )(input_data)
    z = tf.concat([z, input_noise], axis=-1)

    gen_mlp_first_layer_units = result_config['gen_mlp_first_layer_units']
    gen_mlp_layer_units = []
    for i in range(gen_mlp_first_layer_units, -1, -1):
        gen_mlp_layer_units.append(2 ** i)

    z = get_mlp_block(
        input_shape=(2 ** result_config['gen_lstm_units'] + result_config['gen_noise_shape'],),
        layer_units=gen_mlp_layer_units,
        dropout=result_config['gen_mlp_dropout'],
        activation=gan_config.ACTIVATIONS[result_config['gen_mlp_idx_activation']],
        last_activation=None
    )(z)
    generator = Model(inputs=[input_data, input_noise], outputs=[z], name='generator')

    # discriminator
    input_data = Input(shape=(result_config['gen_input_timesteps'] + 1, 1))
    x = LSTM(
        units=2 ** result_config['dis_lstm_units'],
        dropout=result_config['dis_lstm_dropout'],
        recurrent_dropout=result_config['dis_lstm_recurrent_dropout'],
        return_sequences=False
    )(input_data)

    dis_mlp_first_layer_units = result_config['dis_mlp_first_layer_units']
    dis_mlp_layer_units = []
    for i in range(dis_mlp_first_layer_units, -1, -1):
        dis_mlp_layer_units.append(2 ** i)

    x = get_mlp_block(
        input_shape=(2 ** result_config['dis_lstm_units'],),
        layer_units=dis_mlp_layer_units,
        dropout=result_config['dis_mlp_dropout'],
        activation=gan_config.ACTIVATIONS[result_config['dis_mlp_idx_activation']],
        last_activation='sigmoid'
    )(x)
    discriminator = Model(inputs=[input_data], outputs=[x])
    return generator, discriminator


def train(generator, discriminator, result_config):
    input_timesteps = result_config['gen_input_timesteps']

    # load data
    df = read_csv(DATA_FILE, header=DATA_HEADER)
    data = df.iloc[:, DATA_COLUMN].values.reshape(-1, 1)
    data_obj = Data(data, SPLIT_RATIO, input_timesteps=input_timesteps)

    X_gen_train = data_obj.X_train
    y_gen_train = data_obj.y_train
    X_gen_test = data_obj.X_test
    y_gen_test = data_obj.y_test
    X_dis_train, _ = data_obj.create_dataset(input_timesteps=input_timesteps + 1, data=data_obj.scale_data_train)
    X_dis_test, _ = data_obj.create_dataset(input_timesteps=input_timesteps + 1, data=data_obj.scale_data_test)

    # get fixed config
    fixed_config = gan_config.PSO['fixed_config']

    noise_size = result_config['gen_noise_shape']

    # get optimizer
    gen_optimizer = tf.optimizers.get('adam')
    dis_optimizer = tf.optimizers.get('adam')

    # get generator loss function
    w_gan = result_config['gen_w_gan']
    w_reg = result_config['gen_w_reg']
    w_direct = result_config['gen_w_direct']
    threshold = fixed_config['gen_threshold']
    generator_loss_fn = get_generator_loss_function(w_gan, w_reg, w_direct, threshold)

    # split train-validation
    validation_split = fixed_config['validation_split']
    n_trains = int(X_gen_train.shape[0] * (1 - validation_split))
    X_gen_train_true = X_gen_train[:n_trains]
    y_gen_train_true = y_gen_train[:n_trains]
    X_gen_val = X_gen_train[n_trains:]
    y_gen_val = y_gen_train[n_trains:]

    batch_size = fixed_config['batch_size']
    n_epochs = fixed_config['epochs']
    prediction_times = fixed_config['prediction_times']
    num_samples = min(len(X_gen_train_true), len(X_dis_train))
    num_batches = ceil(num_samples / batch_size)

    early_stopping = fixed_config['early_stopping']
    best_gen_validation = float('inf')
    best_gen_weights = None
    not_improve = 0
    validation_error = 0

    # training
    for epoch in range(n_epochs):
        for idx_batch in tf.range(num_batches):
            # data
            idx_start = idx_batch * batch_size
            idx_end = idx_start + batch_size
            if idx_end > num_samples:
                idx_end = num_samples
            X_gen_batch = X_gen_train_true[idx_start:idx_end]
            y_gen_batch = y_gen_train_true[idx_start:idx_end]
            X_dis_batch = X_dis_train[idx_start:idx_end]

            # feed forward
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                y_gen_pred = generator([X_gen_batch, tf.random.normal(shape=(X_gen_batch.shape[0], noise_size))],
                                       training=True)
                tmp = tf.reshape(y_gen_pred, [y_gen_pred.shape[0], y_gen_pred.shape[1], 1])
                tmp = tf.concat([X_gen_batch, tmp], axis=1)
                fake_output = discriminator(tmp, training=True)
                real_output = discriminator(X_dis_batch, training=True)

                gen_total_loss, _, _, _ = generator_loss_fn(
                    X_gen_batch, y_gen_batch, y_gen_pred, fake_output
                )
                dis_loss = discriminator_loss(real_output, fake_output)

            # back prop
            gradients_of_gen = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
            gradients_of_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
            dis_optimizer.apply_gradients(zip(gradients_of_dis, discriminator.trainable_variables))

        validation_error, _ = evaluate_model(generator, X_gen_val, y_gen_val, noise_size, prediction_times)
        if validation_error < best_gen_validation:
            best_gen_validation = validation_error
            best_gen_weights = generator.get_weights()
            not_improve = 0
        else:
            not_improve += 1

        if not_improve > early_stopping:
            generator.set_weights(best_gen_weights)
            break

    # Evaluating
    _, y_gen_test_pred = evaluate_model(generator, X_gen_test, y_gen_test, noise_size, prediction_times)
    y_gen_test_inv = data_obj.invert_transform(y_gen_test)
    y_gen_test_pred_inv = data_obj.invert_transform(y_gen_test_pred)
    test_error_inv = np.sqrt(mean_squared_func(y_gen_test_inv, y_gen_test_pred_inv))
    # print('training gan done, rmse after invert transform: {}'.format(test_loss_inv))

    fitness_value = validation_error
    return fitness_value, generator, test_error_inv


def predict(args):
    model, X, noise_size = args
    return model([X, tf.random.normal(shape=(X.shape[0], noise_size))], training=False).numpy()


def evaluate_model(model, X, y, noise_size, prediction_times):
    results = []
    # args = [(model, X, noise_size)] * prediction_times
    # results = multiprocessing.pool.ThreadPool().map(predict, args)
    for i in range(prediction_times):
        results.append(predict((model, X, noise_size)))
    y_pred_final = np.stack(results).mean(axis=0)
    return mean_squared_func(y, y_pred_final), y_pred_final


def fitness_function(result_config):  # result config: config from pso search
    generator, discriminator = build_model(result_config)
    return train(generator, discriminator, result_config)

