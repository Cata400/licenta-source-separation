import tensorflow as tf
import numpy as np


def create_model():
    input = tf.keras.layers.Input(shape=(100, 1))
    h1 = tf.keras.layers.Dense(10)(input)
    out = tf.keras.layers.Dense(10)(h1)

    model = tf.keras.models.Model(inputs=input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    model.summary()

    return model


x = np.random.rand(100, 1)
x_val = np.random.rand(32, 1)
y = x ** 2
y_val = x_val ** 2

model = create_model()
model.fit(x, y, batch_size=16, epochs=100, validation_data=(x_val, y_val),
          callbacks=tf.keras.callbacks.ModelCheckpoint('test.h5', verbose=1, save_best_only=True))

model2 = tf.keras.models.load_model('test.h5')
print(model2.summary())
