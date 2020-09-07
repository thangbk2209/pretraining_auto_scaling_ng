import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time

RUN_ID = time.strftime('%Y_%m_%d-%H_%M_%S')

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(CORE_DATA_DIR, 'models', RUN_ID)
CONFIGS_DIR = os.path.join(CORE_DATA_DIR, 'configs')
GEN_AE_LOG_DIR = os.path.join(CORE_DATA_DIR, 'logs', 'gen_ae', RUN_ID)
DIS_AE_LOG_DIR = os.path.join(CORE_DATA_DIR, 'logs', 'dis_ae', RUN_ID)
PLOT_RESULT_DIR = os.path.join(CORE_DATA_DIR, 'plot_results')


DATA_FILE = os.path.join(CORE_DATA_DIR, 'input_data', 'google_trace', '1_job', '5_mins.csv')
DATA_COLUMN = 3
DATA_HEADER = None
SPLIT_RATIO = (0.8, 0.2)
INPUT_TIMESTEPS = 32
PREDICTION_TIMES_TRAIN = 50
PREDICTION_TIMES_EVALUATE = 300
LOAD_PRETRAINED_MODELS_FROM = os.path.join(MODELS_DIR, '2020_09_04-09_22_32')

gen_tensorboard_cb = TensorBoard(GEN_AE_LOG_DIR)
dis_tensorboard_cb = TensorBoard(DIS_AE_LOG_DIR)
early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True)


class GanConfig:
    GAN = {
        'noise_size': 2,
        'w_gan': 0.3,
        'w_reg': 0.5,
        'w_direct': 0.2,
        'threshold': 0.08,
        'gen_optimizer': 'adam',  # or tf.keras.optimizers.Adam(learning_rate=0.001)
        'dis_optimizer': 'adam',
        'batch_size': 512,
        'epochs': 100
    }

    GEN_AE = {
        'input_shape': (INPUT_TIMESTEPS, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'loss': 'mse',
        'optimizer': 'adam',
        'batch_size': 256,
        'epochs': 200,
        'validation_split': 0.2,
        'callbacks': [gen_tensorboard_cb, early_stopping_cb],
    }

    GEN_MLP = {
        'layer_units': [64, 32, 1],
        'dropout': 0.1,
        'activation': 'tanh',
        'last_activation': None
    }

    DIS_AE = {
        'input_shape': (INPUT_TIMESTEPS + 1, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'loss': 'mse',
        'optimizer': 'adam',
        'batch_size': 256,
        'epochs': 200,
        'validation_split': 0.2,
        'callbacks': [dis_tensorboard_cb, early_stopping_cb],
    }

    DIS_MLP = {
        'layer_units': [64, 32, 1],
        'dropout': 0.1,
        'activation': 'tanh',
        'last_activation': 'sigmoid'
    }
