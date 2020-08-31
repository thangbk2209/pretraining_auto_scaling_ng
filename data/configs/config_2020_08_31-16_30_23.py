import os
import tensorflow as tf
import time


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(CORE_DATA_DIR, 'models')
CONFIGS_DIR = os.path.join(CORE_DATA_DIR, 'configs')

RUN_ID = time.strftime('%Y_%m_%d-%H_%M_%S')
DATA_FILE = os.path.join(CORE_DATA_DIR, 'input_data', 'google_trace', '1_job', '5_mins.csv')
DATA_COLUMN = 3
DATA_HEADER = None
SPLIT_RATIO = (0.8, 0.2)
INPUT_TIMESTEPS = 16
PREDICTION_TIMES = 200
METRIC_GENERATOR_LOSS = 'RootMeanSquaredError'
# metric for monitoring training loss, testing loss of generator
# see tf.keras.metrics.get()


class GanConfig:
    GAN = {
        'noise_shape': 4,
        'w_gan': 1.,
        'w_reg': 1.,
        'w_direct': 1.,
        'threshold': 0.01,
        'gen_optimizer': 'adam',  # or tf.keras.optimizers.Adam(learning_rate=0.001)
        'dis_optimizer': 'adam',
        'batch_size': 512,
        'epochs': 1
    }

    GEN_AE = {
        'input_shape': (INPUT_TIMESTEPS, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'loss': 'mse',
        'optimizer': 'adam',
        'batch_size': 256,
        'epochs': 2,
        'validation_split': 0.1,
        'callbacks': None,
    }

    GEN_MLP = {
        'layer_units': [64, 32, 1],
        'dropout': None,
        'activation': 'tanh',
        'last_activation': None
    }

    DIS_AE = {
        'input_shape': (INPUT_TIMESTEPS + 1, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'loss': 'mse',
        'optimizer': 'adam',
        'batch_size': 256,
        'epochs': 2,
        'validation_split': 0.1,
        'callbacks': None,
    }

    DIS_MLP = {
        'layer_units': [64, 32, 1],
        'dropout': None,
        'activation': 'tanh',
        'last_activation': 'sigmoid'
    }
