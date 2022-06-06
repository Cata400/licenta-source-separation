from imports import *
from custom_losses import *
from custom_layers import *


def create_model_open_unmix_1_source(input_shape, optimizer, loss, n_fft, max_freq, sr, batch_size, mean, std):
    input = Input(shape=input_shape)
    if max_freq != sr / 2:
        limit = int(np.ceil(max_freq / (sr / 2) * (n_fft // 2 + 1)))
        cropped = Lambda(lambda x: x[:, :limit, :])(input)
        mean_in = mean[:limit]
        std_in = std[:limit]
    else:
        cropped = input
        mean_in = mean
        std_in = std

    scaled_in = ScaleInLayer(mean=mean_in, std=std_in, batch_size=batch_size, name='ScaleIn')(cropped)
    scaled_in = Reshape((scaled_in.shape[-1], scaled_in.shape[-2]))(scaled_in)

    fc1 = Dense(units=n_fft // 8)(scaled_in)
    bn1 = BatchNormalization()(fc1)
    tanh = Activation('tanh')(bn1)

    blstm1 = Bidirectional(LSTM(n_fft // 16, return_sequences=True))(tanh)
    blstm2 = Bidirectional(LSTM(n_fft // 16, return_sequences=True))(blstm1)
    blstm3 = Bidirectional(LSTM(n_fft // 16, return_sequences=True))(blstm2)

    concat = Concatenate(axis=-1)([tanh, blstm3])
    fc2 = Dense(units=n_fft // 8)(concat)
    bn2 = BatchNormalization()(fc2)
    relu = Activation('relu')(bn2)

    fc3 = Dense(units=n_fft // 2 + 1)(relu)
    bn3 = BatchNormalization()(fc3)
    bn3 = Reshape((bn3.shape[-1], bn3.shape[-2]))(bn3)

    scaled_out = ScaleOutLayer(mean=mean, std=std, batch_size=batch_size, name='ScaleOut')(bn3)

    mask = Activation('relu')(scaled_out)
    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    return model


def create_model_u_net_1_source(input_shape, optimizer, loss, initial_filters, stride, kernel_size, drop_out):
    input = Input(shape=input_shape)
    print(input.shape)

    # Encoder
    conv1 = Conv2D(filters=initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=0.2)(bn1)
    print(act1.shape)

    conv2 = Conv2D(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=0.2)(bn2)
    print(act2.shape)

    conv3 = Conv2D(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=0.2)(bn3)
    print(act3.shape)

    conv4 = Conv2D(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act3)
    bn4 = BatchNormalization()(conv4)
    act4 = LeakyReLU(alpha=0.2)(bn4)
    print(act4.shape)

    conv5 = Conv2D(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = LeakyReLU(alpha=0.2)(bn5)
    print(act5.shape)

    conv6 = Conv2D(filters=32 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act5)
    bn6 = BatchNormalization()(conv6)
    act6 = LeakyReLU(alpha=0.2)(bn6)
    print(act6.shape)

    # ##############
    # conv7 = Conv2D(filters=64 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act6)
    # bn7 = BatchNormalization()(conv7)
    # act7 = LeakyReLU(alpha=0.2)(bn7)
    # print(act7.shape)
    #
    #
    # deconv6 = Conv2DTranspose(filters=32 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act7)
    # bn6_2 = BatchNormalization()(deconv6)
    # act6_2 = Activation('relu')(bn6_2)
    # act6_2 = Dropout(rate=drop_out)(act6_2)
    # dec6 = Concatenate(axis=-1)([act6, act6_2])
    # print(act6_2.shape, dec6.shape)
    # ##############

    # Decoder
    deconv5 = Conv2DTranspose(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        act6)
    bn5_2 = BatchNormalization()(deconv5)
    act5_2 = Activation('relu')(bn5_2)
    act5_2 = Dropout(rate=drop_out)(act5_2)
    dec5 = Concatenate(axis=-1)([act5, act5_2])
    print(act5_2.shape, dec5.shape)

    # deconv4 = Conv2DTranspose(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer='l2')(dec5)
    deconv4 = Conv2DTranspose(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec5)
    bn4_2 = BatchNormalization()(deconv4)
    act4_2 = Activation('relu')(bn4_2)
    act4_2 = Dropout(rate=drop_out)(act4_2)
    dec4 = Concatenate(axis=-1)([act4, act4_2])
    print(act4_2.shape, dec4.shape)

    deconv3 = Conv2DTranspose(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec4)
    bn3_2 = BatchNormalization()(deconv3)
    act3_2 = Activation('relu')(bn3_2)
    act3_2 = Dropout(rate=drop_out)(act3_2)
    dec3 = Concatenate(axis=-1)([act3, act3_2])
    print(act3_2.shape, dec3.shape)

    deconv2 = Conv2DTranspose(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec3)
    bn2_2 = BatchNormalization()(deconv2)
    act2_2 = Activation('relu')(bn2_2)
    dec2 = Concatenate(axis=-1)([act2, act2_2])
    print(act2_2.shape, dec2.shape)

    deconv1 = Conv2DTranspose(filters=initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(dec2)
    bn1_2 = BatchNormalization()(deconv1)
    act1_2 = Activation('relu')(bn1_2)
    dec1 = Concatenate(axis=-1)([act1, act1_2])
    print(act1_2.shape, dec1.shape)

    mask = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=stride, padding='same')(dec1)
    mask = BatchNormalization()(mask)
    mask = Activation('sigmoid')(mask)
    print(mask.shape)

    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=L11_norm)
    model.summary()

    return model


def create_model_u_net_1_source_maxpooling(input_shape, optimizer, loss, initial_filters, stride, kernel_size,
                                           drop_out):
    input = Input(shape=input_shape)
    print(input.shape)

    # Encoder
    conv1 = Conv2D(filters=initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=0.2)(bn1)
    print(act1.shape)

    # conv2 = Conv2D(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act1)
    # bn2 = BatchNormalization()(conv2)
    # act2 = LeakyReLU(alpha=0.2)(bn2)
    # print(act2.shape)
    act2 = MaxPool2D((2, 2))(act1)
    print(act2.shape)

    conv3 = Conv2D(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=0.2)(bn3)
    print(act3.shape)

    # conv4 = Conv2D(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act3)
    # bn4 = BatchNormalization()(conv4)
    # act4 = LeakyReLU(alpha=0.2)(bn4)
    # print(act4.shape)
    act4 = MaxPool2D((2, 2))(act3)
    print(act4.shape)

    conv5 = Conv2D(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = LeakyReLU(alpha=0.2)(bn5)
    print(act5.shape)

    # conv6 = Conv2D(filters=32 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act5)
    # bn6 = BatchNormalization()(conv6)
    # act6 = LeakyReLU(alpha=0.2)(bn6)
    # print(act6.shape)
    act6 = MaxPool2D((2, 2))(act5)
    print(act6.shape)

    # ##############
    # conv7 = Conv2D(filters=64 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act6)
    # bn7 = BatchNormalization()(conv7)
    # act7 = LeakyReLU(alpha=0.2)(bn7)
    # print(act7.shape)
    #
    #
    # deconv6 = Conv2DTranspose(filters=32 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(act7)
    # bn6_2 = BatchNormalization()(deconv6)
    # act6_2 = Activation('relu')(bn6_2)
    # act6_2 = Dropout(rate=drop_out)(act6_2)
    # dec6 = Concatenate(axis=-1)([act6, act6_2])
    # print(act6_2.shape, dec6.shape)
    # ##############

    # Decoder
    deconv5 = Conv2DTranspose(filters=16 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        act6)
    bn5_2 = BatchNormalization()(deconv5)
    act5_2 = Activation('relu')(bn5_2)
    act5_2 = Dropout(rate=drop_out)(act5_2)
    dec5 = Concatenate(axis=-1)([act5, act5_2])
    print(act5_2.shape, dec5.shape)

    # deconv4 = Conv2DTranspose(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer='l2')(dec5)
    deconv4 = Conv2DTranspose(filters=8 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec5)
    bn4_2 = BatchNormalization()(deconv4)
    act4_2 = Activation('relu')(bn4_2)
    act4_2 = Dropout(rate=drop_out)(act4_2)
    dec4 = Concatenate(axis=-1)([act4, act4_2])
    print(act4_2.shape, dec4.shape)

    deconv3 = Conv2DTranspose(filters=4 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec4)
    bn3_2 = BatchNormalization()(deconv3)
    act3_2 = Activation('relu')(bn3_2)
    act3_2 = Dropout(rate=drop_out)(act3_2)
    dec3 = Concatenate(axis=-1)([act3, act3_2])
    print(act3_2.shape, dec3.shape)

    deconv2 = Conv2DTranspose(filters=2 * initial_filters, kernel_size=kernel_size, strides=stride, padding='same')(
        dec3)
    bn2_2 = BatchNormalization()(deconv2)
    act2_2 = Activation('relu')(bn2_2)
    dec2 = Concatenate(axis=-1)([act2, act2_2])
    print(act2_2.shape, dec2.shape)

    deconv1 = Conv2DTranspose(filters=initial_filters, kernel_size=kernel_size, strides=stride, padding='same',
                              kernel_regularizer='l2')(dec2)
    bn1_2 = BatchNormalization()(deconv1)
    act1_2 = Activation('relu')(bn1_2)
    dec1 = Concatenate(axis=-1)([act1, act1_2])
    print(act1_2.shape, dec1.shape)

    mask = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer='l2')(
        dec1)
    mask = BatchNormalization()(mask)
    mask = Activation('sigmoid')(mask)
    print(mask.shape)

    output = Multiply()([input, mask])

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=L11_norm)
    model.summary()

    return model


def create_model_cdae_1_source(input_shape, optimizer, loss):
    input = Input(shape=input_shape)
    print(input.shape)

    input_reshaped = Reshape((input_shape[1], input_shape[0], input_shape[2]))(input)
    print(input_reshaped.shape)

    # ENCODER
    conv1 = Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_reshaped)
    conv1 = Activation('relu')(conv1)
    pooling1 = MaxPool2D(pool_size=(3, 5))(conv1)
    print(conv1.shape)
    print(pooling1.shape)

    conv2 = Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), padding='same')(pooling1)
    conv2 = Activation('relu')(conv2)
    pooling2 = MaxPool2D(pool_size=(1, 5))(conv2)
    print(conv2.shape)
    print(pooling2.shape)

    conv3 = Conv2D(filters=30, kernel_size=(3, 3), strides=(1, 1), padding='same')(pooling2)
    conv3 = Activation('relu')(conv3)
    print(conv3.shape)

    conv4 = Conv2D(filters=40, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv3)
    conv4 = Activation('relu')(conv4)
    print(conv4.shape)

    # DECODER
    conv5 = Conv2D(filters=30, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv4)
    conv5 = Activation('relu')(conv5)
    print(conv5.shape)

    conv6 = Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv5)
    conv6 = Activation('relu')(conv6)
    print(conv6.shape)

    upconv6 = UpSampling2D(size=(1, 5))(conv6)
    print(upconv6.shape)

    conv7 = Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv6)
    conv7 = Activation('relu')(conv7)
    upconv7 = UpSampling2D(size=(3, 5))(conv7)
    print(conv7.shape)
    print(upconv7.shape)

    output = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv7)
    output = Activation('relu')(output)
    print(output.shape)

    output = Reshape(input_shape)(output)
    print(output.shape)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=L2_norm)
    model.summary()

    return model


def create_model_wavenet(input_shape):
    n_filters = 128  # 32, 8, 16
    filter_width = 1
    dilation_rates = [2 ** i for i in range(10)] * 3  # 10
    drop_out = 0.25

    input = Input(shape=input_shape, batch_size=1, dtype=tf.float32)
    x = input
    y = Flatten()(x)
    skips = []

    x = Conv1D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(x)
    for dilation_rate in dilation_rates:
        x_f = Conv1D(filters=n_filters, kernel_size=filter_width, padding='same', dilation_rate=dilation_rate)(x)
        x_g = Conv1D(filters=n_filters, kernel_size=filter_width, padding='same', dilation_rate=dilation_rate)(x)
        z = Multiply()([Activation('tanh')(x_f), Activation('sigmoid')(x_g)])

        z = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation='relu')(z)
        x = Add()([x, z])

        skips.append(z)

    out = Activation('relu')(Add()(skips))

    out = Conv1D(filters=2048, kernel_size=3, padding='same')(out)  # 128
    out = Activation('relu')(out)

    out = Conv1D(filters=256, kernel_size=3, padding='same')(out)
    out = Activation('relu')(out)

    out = Conv1D(filters=1, kernel_size=1, padding='same')(out)
    out = tf.stack([out, input])

    print(out.shape)

    model = tf.keras.models.Model(inputs=input, outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(optimizer, loss=custom_loss2)
    model.summary()

    return model
