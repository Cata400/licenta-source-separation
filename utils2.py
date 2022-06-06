import numpy as np
from scipy.signal import windows
import tensorflow as tf


def sigwin(x, window_length, w_type, overlap):
    overlap = overlap / 100
    w = []
    delay = int((1 - overlap) * window_length)

    if w_type != 'rect':
        win = windows.get_window(w_type, window_length)

    for i in range(0, len(x), delay):
        if i + window_length <= len(x):
            if w_type == 'rect':
                w.append(x[i:i + window_length])
            else:
                w.append(np.multiply(win, x[i:i + window_length]))

    return np.asanyarray(w, dtype='float32')


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))


def serialize_data_1_source(mixture, source):
    feat_dict = {
                'mixture': bytes_feature(mixture),
                'source': bytes_feature(source)
    }

    ftexample = tf.train.Example(features=tf.train.Features(feature=feat_dict))
    ftserialized = ftexample.SerializeToString()
    return ftserialized


def parse_and_decode_1_source(example_proto):
    feature_description = {
        'mixture': tf.io.FixedLenFeature([], tf.string),
        'source': tf.io.FixedLenFeature([], tf.string),
    }
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_mixture = tf.io.parse_tensor(element['mixture'], 'float32')
    decoded_source = tf.io.parse_tensor(element['source'], 'float32')

    return decoded_mixture, decoded_source