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


def train_autoencoder(X_train, y_train, X_val, y_val, X_test, y_test):
    autoencoder = LSTMAutoEncoder(
        input_shape=(Config.INPUT_TIME_STEPS, Config.FEATURES),
        layer_units_encoder=Config.AE_CONFIG['layer_units_encoder'],
        layer_units_decoder=Config.AE_CONFIG['layer_units_decoder'],
        timesteps_decoder=Config.AE_CONFIG['time_steps_decoder'],
        drop_out=Config.AE_CONFIG['drop_out'],
        recurrent_drop_out=Config.AE_CONFIG['recurrent_drop_out'],
        activation=Config.AE_CONFIG['activation'],
        recurrent_activation=Config.AE_CONFIG['recurrent_activation']
    )
    autoencoder.summary()
    # autoencoder.plot_model(save_dir=os.path.join(PLOT_MODELS_DIR, 'autoencoder.png'))
    autoencoder.model.compile(
        optimizer=Config.AE_CONFIG['optimizer'],
        loss=Config.AE_CONFIG['loss']
    )

    tensorboard_cb = TensorBoard(AE_LOG_DIR)
    early_stopping_cb = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    history = autoencoder.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.AE_CONFIG['batch_size'],
        epochs=Config.AE_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )
    print(f'mse test: {autoencoder.model.evaluate(X_test, y_test)}')
    print(f'y_test[1]: \n{y_test[1]}')
    print(f'y_ped[1]: \n{autoencoder.model.predict(X_test[1:2])}')

    print('saving models')
    autoencoder.model.save(filepath=os.path.join(MODELS_DIR, 'autoencoder.h5'))
    autoencoder.encoder.save(filepath=os.path.join(MODELS_DIR, 'encoder.h5'))
    print(f'saved to {MODELS_DIR}')
    return autoencoder


def train_mlp(encoder, X_train, y_train, X_val, y_val, X_test, y_test):
    mlp_net = MLPNet(
        input_shape=encoder.output_shape[1],
        hidden_layer_units=Config.MLP_CONFIG['hidden_layer_units'],
        hidden_activation=Config.MLP_CONFIG['hidden_activation'],
        drop_out=Config.MLP_CONFIG['drop_out']
    )
    model = Sequential([encoder, mlp_net.model])
    encoder.trainable = False
    model.compile(
        optimizer=Config.MLP_CONFIG['optimizer'],
        loss=Config.MLP_CONFIG['loss'],
    )

    tensorboard_cb = TensorBoard(MLP_LOG_DIR)
    early_stopping_cb = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.MLP_CONFIG['batch_size'],
        epochs=Config.MLP_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )

    model.summary()
    print('saving models')
    model.save(os.path.join(MODELS_DIR, 'global_model.h5'))
    mlp_net.model.save(os.path.join(MODELS_DIR, 'mlp_net.h5'))
    print(f'saved to {MODELS_DIR}')

    return model


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

    print('Training AutoEncoder: ')
    # data for autoencoder
    input_cols = [i for i in range(len(INPUT_COLS))]
    X_train, y_train = data_obj.create_dataset(
        type_dataset='train',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.AE_CONFIG['time_steps_decoder'],
        input_cols=input_cols, predict_cols=input_cols)
    X_val, y_val = data_obj.create_dataset(
        type_dataset='validation',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.AE_CONFIG['time_steps_decoder'],
        input_cols=input_cols, predict_cols=input_cols)
    X_test, y_test = data_obj.create_dataset(
        type_dataset='test',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.AE_CONFIG['time_steps_decoder'],
        input_cols=input_cols, predict_cols=input_cols)
    print('dataset shape:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    autoencoder = train_autoencoder(X_train, y_train, X_val, y_val, X_test, y_test)

    # data for global model
    print('Training global model: ')
    input_cols = [i for i in range(len(INPUT_COLS))]
    predict_cols = []
    for it in PREDICT_COLS:
        for i in range(len(INPUT_COLS)):
            if it == INPUT_COLS[i]:
                predict_cols.append(i)
    X_train, y_train = data_obj.create_dataset(
        type_dataset='train',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=predict_cols)
    X_val, y_val = data_obj.create_dataset(
        type_dataset='validation',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=predict_cols)
    X_test, y_test = data_obj.create_dataset(
        type_dataset='test',
        input_time_steps=Config.INPUT_TIME_STEPS, predict_time_steps=Config.PREDICT_TIME_STEPS,
        input_cols=input_cols, predict_cols=predict_cols)
    print('dataset shape:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    # train mlp
    global_model = train_mlp(autoencoder.encoder, X_train, y_train, X_val, y_val, X_test, y_test)

    test_err = global_model.evaluate(X_test, y_test)
    print('mse test: {:.4f}'.format(test_err))
    print('rmse test: {:.4f}'.format(np.sqrt(test_err)))

    y_ped = global_model.predict(X_test)
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







