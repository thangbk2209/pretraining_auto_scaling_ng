import os
import tensorflow as tf

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')


class GanConfig:
    GAN = {
        'noise_shape': 4,
        'w_gan': 1.,
        'w_reg': 1.,
        'w_direct': 1.,
        'threshold': 0.01,
        'gen_optimizer': 'adam',  # or tf.optimizers.Adam(learning_rate=0.001)
        'dis_optimizer': 'adam',
        'batch_size': 512,
        'epochs': 1
    }

    GEN_AE = {
        'input_shape': (16, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'drop_out': 0.0,
        'recurrent_drop_out': 0.0,
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
        'input_shape': (GEN_AE['input_shape'][0] + 1, 1),
        'layer_units_encoder': [64, 8],
        'layer_units_decoder': [64, 8],
        'timesteps_decoder': 8,
        'drop_out': 0.0,
        'recurrent_drop_out': 0.0,
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
