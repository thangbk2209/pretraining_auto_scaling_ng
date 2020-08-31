import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
binary_crossentropy = BinaryCrossentropy()


def discriminator_loss(real_dis_output, fake_dis_output):
    true = tf.concat([tf.ones_like(real_dis_output), tf.zeros_like(fake_dis_output)], axis=0)
    predict = tf.concat([real_dis_output, fake_dis_output], axis=0)
    return binary_crossentropy(true, predict)


def generator_bce_loss(fake_dis_output):
    return binary_crossentropy(tf.ones_like(fake_dis_output), fake_dis_output)


def generator_direct_loss(X, y_gen_true, y_gen_pred):
    tmp = tf.sign(y_gen_true - X[:, -1, :]) - tf.sign(y_gen_pred - X[:, -1, :])
    tmp = tf.abs(tmp) / 2.
    return tf.reduce_mean(tmp)


def generator_mse_regression_loss(y_gen_true, y_gen_pred):
    return tf.reduce_mean(tf.square(y_gen_true - y_gen_pred))


def get_generator_loss_function(w_gan, w_reg, w_direct, threshold):
    def loss_function(X, y_gen_true, y_gen_pred, fake_dis_output):
        gan_loss = generator_bce_loss(fake_dis_output)
        reg_loss = generator_mse_regression_loss(y_gen_true, y_gen_pred)
        direct_loss = generator_direct_loss(X, y_gen_true, y_gen_pred)
        if reg_loss < threshold:
            total_loss = gan_loss
        else:
            total_loss = w_gan*gan_loss + w_reg*reg_loss + w_direct*direct_loss
        return total_loss, gan_loss, reg_loss, direct_loss
    return loss_function

