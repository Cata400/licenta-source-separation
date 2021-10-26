from imports import *


def create_model_open_unmix_1_source(input_shape, optimizer, loss, n_fft, max_freq, sr):
    input = Input(shape=input_shape)
    if max_freq != sr / 2:
        cropped = Lambda(lambda x: x[:, :int(np.ceil(max_freq / (sr / 2) * (n_fft // 2 + 1))), :])(input)
    else:
        cropped = input

    input_mean = tf.math.reduce_mean(cropped, axis=-1)
    input_mean = tf.expand_dims(input_mean, axis=-1)
    input_std = tf.math.reduce_std(cropped, axis=-1)
    input_std = tf.expand_dims(input_std, axis=-1)

    scaled_in = tf.math.divide(tf.math.subtract(cropped, input_mean), input_std)
    scaled_in = Reshape((scaled_in.shape[-1], scaled_in.shape[-2]))(scaled_in)

    fc1 = Dense(units=n_fft // 4)(scaled_in)
    bn1 = BatchNormalization()(fc1)
    tanh = Activation('tanh')(bn1)

    blstm1 = Bidirectional(tf.keras.layers.LSTM(n_fft // 4, return_sequences=True))(tanh)
    blstm2 = Bidirectional(tf.keras.layers.LSTM(n_fft // 4, return_sequences=True))(blstm1)
    blstm3 = Bidirectional(tf.keras.layers.LSTM(n_fft // 4, return_sequences=True))(blstm2)

    concat = Concatenate(axis=-1)([tanh, blstm3])
    fc2 = Dense(units=n_fft // 4)(concat)
    bn2 = BatchNormalization()(fc2)
    relu = Activation('relu')(bn2)

    fc3 = Dense(units=n_fft // 2 + 1)(relu)
    bn3 = BatchNormalization()(fc3)
    bn3 = Reshape((bn3.shape[-1], bn3.shape[-2]))(bn3)

    output_std = tf.ones(shape=(1, n_fft // 2 + 1, 1))
    output_mean = tf.ones(shape=(1, n_fft // 2 + 1, 1))

    scaled_out = tf.math.add(tf.math.multiply(bn3, output_std), output_mean)
    mask = Activation('relu')(scaled_out)
    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    return model


def create_model_u_net_1_source(input_shape, optimizer, loss, drop_out):
    pass
