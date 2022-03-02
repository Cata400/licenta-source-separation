import tensorflow.keras.backend as kb


def L11_norm(y_actual, y_pred):
    return kb.sum(kb.abs(y_actual - y_pred))
