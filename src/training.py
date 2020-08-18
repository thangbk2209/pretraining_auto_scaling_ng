from pandas import read_csv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from src.models import LSTMAutoEncoder, MLPNet
from src.preprocessing import Data
import matplotlib.pyplot as plt
import time
from config import *

run_id = time.strftime('%Y_%m_%d-%H_%M_%S')
AE_LOG_DIR = os.path.join(Config.LOG_DIR, 'autoencoder', run_id)
MLP_LOG_DIR = os.path.join(Config.LOG_DIR, 'mlp', run_id)


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
    autoencoder.plot_model(save_dir=Config.PLOT_MODELS_DIR)
    autoencoder.model.compile(
        optimizer=Config.AE_CONFIG['optimizer'],
        loss=Config.AE_CONFIG['loss']
    )
    tensorboard_cb = TensorBoard(AE_LOG_DIR)
    history = autoencoder.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.AE_CONFIG['batch_size'],
        epochs=Config.AE_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb]
    )
    print(f'mse test: {autoencoder.model.evaluate(X_test, y_test)}')
    print(f'y_test[1]: \n{y_test[1]}')
    print(f'y_ped[1]: \n{autoencoder.model.predict(X_test[1:2])}')

    print('saving models')
    autoencoder.model.save(filepath=os.path.join(Config.MODELS_DIR, 'autoencoder.h5'))
    autoencoder.encoder.save(filepath=os.path.join(Config.MODELS_DIR, 'encoder.h5'))
    print(f'saved to {Config.MODELS_DIR}')
    return autoencoder


def train_mlp(encoder, X_train, y_train, X_val, y_val, X_test, y_test, plot_predictions=False):
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
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.MLP_CONFIG['batch_size'],
        epochs=Config.MLP_CONFIG['epochs'],
        verbose=Config.VERBOSE,
        callbacks=[tensorboard_cb]
    )

    model.summary()

    print(f'mse test: {model.evaluate(X_test, y_test)}')
    y_ped = model.predict(X_test)
    plt.plot(y_test.reshape(y_test.shape[0]))
    plt.plot(y_ped.reshape(y_test.shape[0]))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(Config.PLOT_PRED_TRUE_DIR, 'true_pred.png'))

    print('saving models')
    model.save(os.path.join(Config.MODELS_DIR, 'global_model.h5'))
    mlp_net.model.save(os.path.join(Config.MODELS_DIR, 'mlp_net.h5'))
    print(f'saved to {Config.MODELS_DIR}')
    return mlp_net


def run():
    # load data
    data_frame = read_csv(Config.DATA_FILE)
    data = data_frame.iloc[:, Config.INPUT_COLS].values
    if len(Config.INPUT_COLS) == 1:
        data = data.reshape([data.shape[0], 1])

    # create data object
    data_obj = Data(data=data, split_ratio=Config.SPLIT_RATIO, scaler=Config.SCALER)

    print('Training AutoEncoder: ')
    # data for autoencoder
    input_cols = [i for i in range(len(Config.INPUT_COLS))]
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
    input_cols = [i for i in range(len(Config.INPUT_COLS))]
    predict_cols = []
    for it in Config.PREDICT_COLS:
        for i in range(len(Config.INPUT_COLS)):
            if it == Config.INPUT_COLS[i]:
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

    mlp_net = train_mlp(autoencoder.encoder, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    run()



