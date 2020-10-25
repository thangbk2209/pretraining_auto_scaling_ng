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

gen_tensorboard_cb = TensorBoard(GEN_AE_LOG_DIR)
dis_tensorboard_cb = TensorBoard(DIS_AE_LOG_DIR)
early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True)


class GanConfig:
    GAN = {
        'noise_size': 4,
        'w_gan': 0.3,
        'w_reg': 0.5,
        'w_direct': 0.2,
        'threshold': 0.2,
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

    ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'elu']
    # fix 1 layer LSTM
    PSO = {
        'fixed_config': {
            'batch_size': 512,
            'epochs': 1,
            'gen_threshold': 0.1,
            'prediction_times': 2
        },
        'pso_config': {
            'max_iter': 2,
            'step_save': 2,
            'n_particles': 2
        },
        'domain': [
            {'name': 'gen_input_timesteps', 'type': 'discrete', 'domain': [4, 32]},
            {'name': 'gen_lstm_units', 'type': 'discrete', 'domain': [3, 10]}, # encode = 3 -> decode = 2^3 = 8
            {'name': 'gen_lstm_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'gen_lstm_recurrent_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'gen_noise_shape', 'type': 'discrete', 'domain': [1, 10]},
            {'name': 'gen_mlp_first_layer_units', 'type': 'discrete', 'domain': [3, 10]},  # encode = 5 -> [32, 16, 8, 4, 2, 1]
            {'name': 'gen_mlp_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'gen_mlp_idx_activation', 'type': 'discrete', 'domain': [0, 3]},
            {'name': 'gen_w_gan', 'type': 'continuous', 'domain': [0.0, 1.0]},
            {'name': 'gen_w_reg', 'type': 'continuous', 'domain': [0.0, 1.0]},
            {'name': 'gen_w_direct', 'type': 'continuous', 'domain': [0.0, 1.0]},
            # {'name': 'gen_threshold', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'dis_lstm_units', 'type': 'discrete', 'domain': [3, 10]},  # encode = 3 -> decode = 2^3 = 8
            {'name': 'dis_lstm_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'dis_lstm_recurrent_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'dis_mlp_first_layer_units', 'type': 'discrete', 'domain': [3, 10]},  # encode = 5 -> [32, 16, 8, 4, 2, 1]
            {'name': 'dis_mlp_dropout', 'type': 'continuous', 'domain': [0.0, 0.5]},
            {'name': 'dis_mlp_idx_activation', 'type': 'discrete', 'domain': [0, 3]}  # index, 0 -> relu,...
        ]

    }
