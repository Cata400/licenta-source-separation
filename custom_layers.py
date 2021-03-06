import tensorflow as tf


class ScaleInLayer(tf.keras.layers.Layer):
    def __init__(self, mean, std, batch_size, name=None, **kwargs):
        super(ScaleInLayer, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std
        self.batch_size = batch_size

    def build(self, input_shape):
        self.mean_parameter = self.add_weight("mean", shape=(input_shape[-2], 1), trainable=True)
        self.std_parameter = self.add_weight("std", shape=(input_shape[-2], 1), trainable=True)

        self.mean_parameter.assign(self.mean)
        self.std_parameter.assign(self.std)

    def call(self, inputs):
        return tf.math.divide(tf.math.subtract(inputs, self.mean_parameter), self.std_parameter)

    def get_config(self):
        cfg = super(ScaleInLayer, self).get_config()
        cfg['mean'] = self.mean
        cfg['std'] = self.std
        cfg['batch_size'] = self.batch_size
        return cfg


class ScaleOutLayer(tf.keras.layers.Layer):
    def __init__(self, mean, std, batch_size, name=None, **kwargs):
        super(ScaleOutLayer, self).__init__(batch_size=batch_size, name=name, **kwargs)
        self.mean = mean
        self.std = std
        self.batch_size = batch_size

    def build(self, input_shape):
        self.mean_parameter = self.add_weight("mean", shape=(input_shape[-2], 1), trainable=True)
        self.std_parameter = self.add_weight("std", shape=(input_shape[-2], 1), trainable=True)

        self.mean_parameter.assign(tf.zeros(shape=(input_shape[-2], 1)))
        self.std_parameter.assign(tf.ones(shape=(input_shape[-2], 1)))

    def call(self, inputs):
        return tf.math.add(tf.math.multiply(inputs, self.std_parameter), self.mean_parameter)

    def get_config(self):
        cfg = super().get_config()
        cfg['mean'] = self.mean
        cfg['std'] = self.std
        cfg['batch_size'] = self.batch_size
        return cfg
