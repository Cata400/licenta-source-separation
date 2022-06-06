import tensorflow.keras.backend as kb
import tensorflow as tf


def L11_norm(y_actual, y_pred):
    return kb.sum(kb.abs(y_actual - y_pred)) / 128


def L2_norm(y_actual, y_pred):
    return kb.sum((y_actual - y_pred)**2) / 128


def custom_loss2(model):
    def loss(y_true, y_pred):
        y_pred = model.output
        out = y_pred[0]
        i = y_pred[1]
        out = tf.reshape(out, [tf.shape(out)[0], -1])
        i = tf.reshape(i, [tf.shape(i)[0], -1])
        return tf.math.add(tf.math.abs(tf.math.subtract(y_true, out)), tf.math.abs(tf.math.subtract(tf.math.subtract(i, y_true), tf.math.subtract(i, out))))
    return loss
