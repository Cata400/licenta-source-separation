from imports import *
from custom_losses import *
from custom_layers import *


def create_model_open_unmix_1_source(input_shape, optimizer, loss, n_fft, max_freq, sr, batch_size, mean, std):
    input = Input(shape=input_shape)
    if max_freq != sr / 2:
        limit = int(np.ceil(max_freq / (sr / 2) * (n_fft // 2 + 1)))
        cropped = Lambda(lambda x: x[:, :limit, :])(input)
        mean = mean[:limit]
        std = std[:limit]
    else:
        cropped = input

    scaled_in = ScaleInLayer(mean=mean, std=std, batch_size=batch_size, name='ScaleIn')(cropped)
    scaled_in = Reshape((scaled_in.shape[-1], scaled_in.shape[-2]))(scaled_in)

    fc1 = Dense(units=n_fft // 8)(scaled_in)
    bn1 = BatchNormalization()(fc1)
    tanh = Activation('tanh')(bn1)

    blstm1 = Bidirectional(tf.keras.layers.LSTM(n_fft // 16, return_sequences=True))(tanh)
    blstm2 = Bidirectional(tf.keras.layers.LSTM(n_fft // 16, return_sequences=True))(blstm1)
    blstm3 = Bidirectional(tf.keras.layers.LSTM(n_fft // 16, return_sequences=True))(blstm2)

    concat = Concatenate(axis=-1)([tanh, blstm3])
    fc2 = Dense(units=n_fft // 8)(concat)
    bn2 = BatchNormalization()(fc2)
    relu = Activation('relu')(bn2)

    fc3 = Dense(units=n_fft // 2 + 1)(relu)
    bn3 = BatchNormalization()(fc3)
    bn3 = Reshape((bn3.shape[-1], bn3.shape[-2]))(bn3)

    scaled_out = ScaleOutLayer(batch_size=batch_size, name='ScaleOut')(bn3)

    mask = Activation('relu')(scaled_out)
    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    return model


def create_model_u_net_1_source(input_shape, optimizer, loss, initial_filters, stride, kernel_size, drop_out):
    input = Input(shape=(input_shape[0], input_shape[1], 1))
    input = Lambda(lambda x: x[:-1, :, :])(input)

    # Encoder
    conv1 = Conv2D(filters=initial_filters, kernel_size=kernel_size, strides=stride)(input)
    bn1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=0.2)(bn1)

    conv2 = Conv2D(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride)(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=0.2)(bn2)

    conv3 = Conv2D(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride)(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=0.2)(bn3)

    conv4 = Conv2D(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride)(act3)
    bn4 = BatchNormalization()(conv4)
    act4 = LeakyReLU(alpha=0.2)(bn4)

    conv5 = Conv2D(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride)(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = LeakyReLU(alpha=0.2)(bn5)

    conv6 = Conv2D(filters=32 * initial_filters, kernel_size=kernel_size, strides=stride)(act5)
    bn6 = BatchNormalization()(conv6)
    act6 = LeakyReLU(alpha=0.2)(bn6)

    # Decoder
    deconv5 = Conv2DTranspose(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride)(act6)
    bn5_2 = BatchNormalization()(deconv5)
    act5_2 = Activation('relu')(bn5_2)
    act5_2 = Dropout(rate=drop_out)(act5_2)
    dec5 = Concatenate(axis=-1)([act5, act5_2])

    deconv4 = Conv2DTranspose(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride)(dec5)
    bn4_2 = BatchNormalization()(deconv4)
    act4_2 = Activation('relu')(bn4_2)
    act4_2 = Dropout(rate=drop_out)(act4_2)
    dec4 = Concatenate(axis=-1)([act4, act4_2])

    deconv3 = Conv2DTranspose(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride)(dec4)
    bn3_2 = BatchNormalization()(deconv3)
    act3_2 = Activation('relu')(bn3_2)
    act3_2 = Dropout(rate=drop_out)(act3_2)
    dec3 = Concatenate(axis=-1)([act3, act3_2])

    deconv2 = Conv2DTranspose(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride)(dec3)
    bn2_2 = BatchNormalization()(deconv2)
    act2_2 = Activation('relu')(bn2_2)
    dec2 = Concatenate(axis=-1)([act2, act2_2])

    deconv1 = Conv2DTranspose(filters=initial_filters, kernel_size=kernel_size, strides=stride)(dec2)
    bn1_2 = BatchNormalization()(deconv1)
    act1_2 = Activation('relu')(bn1_2)
    dec1 = Concatenate(axis=-1)([act1, act1_2])

    mask = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=stride)(dec1)
    mask = BatchNormalization()(mask)
    mask = Activation('sigmoid')(mask)

    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=L11_norm())
    model.summary()

    return model
