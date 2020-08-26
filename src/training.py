from pandas import read_csv
import os
import shutil
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from src.models import LSTMAutoEncoder, MLPNet
from src.preprocessing import Data
import matplotlib.pyplot as plt
from config import *


def train_autoencoder(encoder_train_inputs, decoder_train_inputs, decoder_train_outputs,
                      encoder_validation_inputs, decoder_validation_inputs, decoder_validation_outputs,
                      encoder_test_inputs, decoder_test_inputs, decoder_test_outputs):
    autoencoder = LSTMAutoEncoder(
        inputs_shape=(Config.INPUT_TIME_STEPS, Config.FEATURES),
        layer_units_encoder=Config.AE_CONFIG['layer_units_encoder'],
        layer_units_decoder=Config.AE_CONFIG['layer_units_decoder'],
        timesteps_decoder=Config.AE_CONFIG['time_steps_decoder'],
        drop_out=Config.AE_CONFIG['drop_out'],
        recurrent_drop_out=Config.AE_CONFIG['recurrent_drop_out'],
        activation=Config.AE_CONFIG['activation'],
        recurrent_activation=Config.AE_CONFIG['recurrent_activation']
    )
    autoencoder.model.summary()
    autoencoder.model.compile(
        optimizer=Config.AE_CONFIG['optimizer'],
        loss=Config.AE_CONFIG['loss']
    )

    tensorboard_cb = TensorBoard(AE_LOG_DIR)
    early_stopping_cb = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    history = autoencoder.model.fit(
        [encoder_train_inputs, decoder_train_inputs], decoder_train_outputs,
        validation_data=([encoder_validation_inputs, decoder_validation_inputs], decoder_validation_outputs),
        batch_size=Config.AE_CONFIG['batch_size'],
        epochs=Config.AE_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )
    print(f'mse test: {autoencoder.model.evaluate([encoder_test_inputs, decoder_test_inputs], decoder_test_outputs)}')
    print(f'y_test[1]: \n{decoder_test_outputs[1]}')
    print(f'y_ped[1]: \n{autoencoder.model.predict([encoder_test_inputs[1:2], decoder_test_inputs[1:2]])}')

    print('saving models')
    autoencoder.model.save(filepath=os.path.join(MODELS_DIR, 'autoencoder.h5'))
    autoencoder.encoder.save(filepath=os.path.join(MODELS_DIR, 'encoder.h5'))
    print(f'saved to {MODELS_DIR}')
    return autoencoder


def train_mlp(encoder, X_train, y_train, X_val, y_val, X_test, y_test):
    mlp = MLPNet(
        hidden_layer_units=Config.MLP_CONFIG['hidden_layer_units'],
        hidden_activation=Config.MLP_CONFIG['hidden_activation'],
        drop_out=Config.MLP_CONFIG['drop_out']
    ).model
    pred_model = Sequential([encoder, mlp])
    encoder.trainable = False
    pred_model.compile(
        optimizer=Config.MLP_CONFIG['optimizer'],
        loss=Config.MLP_CONFIG['loss'],
    )
    pred_model.summary()

    tensorboard_cb = TensorBoard(MLP_LOG_DIR)
    early_stopping_cb = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    pred_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.MLP_CONFIG['batch_size'],
        epochs=Config.MLP_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )

    print('saving models')
    pred_model.save(os.path.join(MODELS_DIR, 'pred_model.h5'))
    print(f'saved to {MODELS_DIR}')

    return pred_model


def run():
    # save config.py
    try:
        shutil.copy(os.path.join(PROJECT_DIR, 'config.py'), os.path.join(CONFIGS_DIR, f'config_{RUN_ID}.py'))
    except Exception as e:
        print(e)

    # create models directory
    try:
        os.mkdir(MODELS_DIR)
    except Exception as e:
        print(e)

    # load data
    data_frame = read_csv(DATA_FILE, header=CSV_HEADER)
    data = data_frame.iloc[:, INPUT_COLS].values
    if len(INPUT_COLS) == 1:
        data = data.reshape(-1, 1)

    # create data object
    data_obj = Data(data=data, split_ratio=Config.SPLIT_RATIO, interval_diff=Config.INTERVAL_DIFF)

    # data
    input_cols = [i for i in range(len(INPUT_COLS))]
    # predict_cols = []
    # for it in PREDICT_COLS:
    #     for i in range(len(INPUT_COLS)):
    #         if it == INPUT_COLS[i]:
    #             predict_cols.append(i)
    X_train, y_train = data_obj.create_dataset(
        type_dataset='train',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=input_cols)
    X_val, y_val = data_obj.create_dataset(
        type_dataset='validation',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=input_cols)
    X_test, y_test = data_obj.create_dataset(
        type_dataset='test',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=input_cols)
    print('dataset shape:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    print('Training AutoEncoder: ')
    decoder_timesteps = Config.AE_CONFIG['time_steps_decoder']
    autoencoder = train_autoencoder(
        X_train, X_train[:, -decoder_timesteps-1:-1, :], X_train[:, -decoder_timesteps:, :],
        X_val, X_val[:, -decoder_timesteps-1:-1, :], X_val[:, -decoder_timesteps:, :],
        X_test, X_test[:, -decoder_timesteps-1:-1, :], X_test[:, -decoder_timesteps:, :],
    )

    print('Training global model: ')
    # train mlp
    pred_model = train_mlp(autoencoder.encoder, X_train, y_train, X_val, y_val, X_test, y_test)
    
    test_err = pred_model.evaluate(X_test, y_test)
    print('mse test: {:.4f}'.format(test_err))
    print('rmse test: {:.4f}'.format(np.sqrt(test_err)))

    y_ped = pred_model.predict(X_test)
    y_pred_invert = data_obj.invert_transform(data_obj.data_test, y_ped.reshape(-1, 1), Config.INPUT_TIME_STEPS)
    y_test_invert = data_obj.invert_transform(data_obj.data_test, y_test.reshape(-1, 1), Config.INPUT_TIME_STEPS)

    print('rmse test after rescaling: {:.4f}'.format(np.sqrt((y_pred_invert - y_test_invert) ** 2).mean()))
    # plot test:
    # all
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_test_invert.reshape(-1))
    plt.plot(y_pred_invert.reshape(-1))
    plt.title('Test mse={:06.2f}'.format(test_err))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(PLOT_PRED_TRUE_DIR, f'true_pred_{RUN_ID}_all.png'))

    # [:100]
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_test_invert.reshape(-1)[:100])
    plt.plot(y_pred_invert.reshape(-1)[:100])
    plt.title('Test mse={:06.2f}, [100:]'.format(test_err))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(PLOT_PRED_TRUE_DIR, f'true_pred_{RUN_ID}_100first.png'))

    # [-100:]
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_test_invert.reshape(-1)[-100:])
    plt.plot(y_pred_invert.reshape(-1)[-100:])
    plt.title('Test mse={:06.2f}, [-100:]'.format(test_err))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(PLOT_PRED_TRUE_DIR, f'true_pred_{RUN_ID}_100last.png'))







