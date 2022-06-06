import numpy as np
import tensorflow as tf


x = np.array([[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]], dtype='float32')
              # [[1, 2, 3], [1, 2, 3], [1, 2, 3]]], dtype='float32')

# x.reshape((x.shape[1], x.shape[2], x.shape[0]))
print(x.shape)

mean = tf.math.reduce_mean(x, axis=-1)
std = tf.math.reduce_std(x, axis=-1)
print(mean.shape)
print(std.shape)
mean = tf.expand_dims(mean, axis=-1)
std = tf.expand_dims(std, axis=-1)
print(mean.shape)
print(std.shape)

# inputs = tf.random.normal([32, 10, 8])
# lstm = tf.keras.layers.LSTM(4)
# output = lstm(inputs)
# print(output.shape)
#
# lstm = tf.keras.layers.LSTM(512, return_sequences=True)
# whole_seq_output = lstm(inputs)
# print(whole_seq_output.shape)
