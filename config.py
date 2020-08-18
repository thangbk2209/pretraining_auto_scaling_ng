import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')


class Config:
    PLOT_MODELS_DIR = os.path.join(CORE_DATA_DIR, 'plot_models')
    PLOT_PRED_TRUE_DIR = os.path.join(CORE_DATA_DIR, 'plot_pred_true')
    MODELS_DIR = os.path.join(CORE_DATA_DIR, 'models')
    LOG_DIR = os.path.join(CORE_DATA_DIR, 'logs')
    DATA_FILE = os.path.join(CORE_DATA_DIR, 'input_data', 'google_trace', 'all_jobs', '5_mins.csv')
    INPUT_COLS, PREDICT_COLS = [0], [0]
    LEARNING_RATE = 3e-4
    SPLIT_RATIO = (0.8, 0.1, 0.1)
    INPUT_TIME_STEPS = 15
    PREDICT_TIME_STEPS = 1
    FEATURES = 1
    VERBOSE = 2
    SCALER_LIST = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
    SCALER = SCALER_LIST[0]

    AE_CONFIG = {
        'layer_units_encoder': [128, 32],
        'layer_units_decoder': [128, 32],
        'time_steps_decoder': 8,
        'drop_out': 0.4,
        'recurrent_drop_out': 0.4,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'batch_size': 256,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 2
    }

    MLP_CONFIG = {
        'hidden_layer_units': [128, 64, 16],
        'drop_out': 0.4,
        'hidden_activation': 'relu',
        'batch_size': 256,
        'optimizer': 'adam',
        'loss': 'mse',
        'epochs': 2
    }
