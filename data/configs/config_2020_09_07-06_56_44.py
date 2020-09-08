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
INPUT_COLS = [3]
PREDICT_TIMES = 300


class Config:
    LEARNING_RATE = 3e-4
    SPLIT_RATIO = (0.7, 0.1, 0.2)
    INPUT_TIMESTEPS = 32
    PREDICT_TIMESTEPS = 1
    FEATURES = len(INPUT_COLS)
    VERBOSE = 2
    INTERVAL_DIFF = 1
    PATIENCE = 20

    AE_CONFIG = {
        'layer_units_encoder': [128, 8],
        'layer_units_decoder': [128, 8],
        'timesteps_decoder': 16,
        'dropout': 0.3,
        'recurrent_dropout': 0.1,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'batch_size': 512,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 500
    }

    MLP_CONFIG = {
        'encoder_trainable': False,
        'hidden_layer_units': [64, 16],
        'dropout': 0.2,
        'hidden_activation': 'tanh',
        'batch_size': 512,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 500
        
    }
