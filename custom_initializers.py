import tensorflow as tf


class TensorInitializer(tf.keras.initializers.Initializer):
    def __init__(self, tensor):
      self.tensor = tensor

    def __call__(self, shape, dtype=None, **kwargs):
      return self.tensor

    def get_config(self):  # To support serialization
      return {"tensor": self.tensor}
