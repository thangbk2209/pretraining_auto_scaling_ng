import os
import time


RUN_ID = time.strftime('%Y_%m_%d-%H_%M_%S')

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')

PLOT_MODELS_DIR = os.path.join(CORE_DATA_DIR, 'plot_models', RUN_ID)
PLOT_PRED_TRUE_DIR = os.path.join(CORE_DATA_DIR, 'plot_pred_true')
MODELS_DIR = os.path.join(CORE_DATA_DIR, 'models', RUN_ID)
LOG_DIR = os.path.join(CORE_DATA_DIR, 'logs')
AE_LOG_DIR = os.path.join(LOG_DIR, 'autoencoder', RUN_ID)
MLP_LOG_DIR = os.path.join(LOG_DIR, 'mlp', RUN_ID)
CONFIGS_DIR = os.path.join(CORE_DATA_DIR, 'configs')
DATA_FILE = os.path.join(CORE_DATA_DIR, 'input_data', 'google_trace', '1_job', '5_mins.csv')
CSV_HEADER = None
INPUT_COLS, PREDICT_COLS = [3], [3]


class Config:
    LEARNING_RATE = 3e-4
    SPLIT_RATIO = (0.7, 0.1, 0.2)
    INPUT_TIME_STEPS = 20
    PREDICT_TIME_STEPS = 1
    FEATURES = len(INPUT_COLS)
    VERBOSE = 2
    INTERVAL_DIFF = 1
    PATIENCE = 10

    AE_CONFIG = {
        'layer_units_encoder': [128, 64],
        'layer_units_decoder': [128, 64],
        'time_steps_decoder': 5,
        'drop_out': 0.2,
        'recurrent_drop_out': 0.2,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'batch_size': 256,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 1
    }

    MLP_CONFIG = {
        'hidden_layer_units': [128, 64, 16],
        'drop_out': 0.2,
        'hidden_activation': 'relu',
        'batch_size': 256,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 1
    }
