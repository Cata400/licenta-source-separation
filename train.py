from utils import *
from models import *



def training(network, train_path, val_path, batch_size, shuffle_buffer_size, input_shape, loss, optimizer,
             drop_out, epochs, callbacks, multiple_sources, n_fft, max_freq, sr, initial_filters, stride, kernel_size):

    train_dataset = tf.data.TFRecordDataset(train_path)
    val_dataset = tf.data.TFRecordDataset(val_path)

    if multiple_sources:
        train_dataset = train_dataset.map(parse_and_decode_all_sources)
        val_dataset = val_dataset.map(parse_and_decode_all_sources)
    else:
        train_dataset = train_dataset.map(parse_and_decode_1_source)
        val_dataset = val_dataset.map(parse_and_decode_1_source)

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

    if not multiple_sources:
        if network.lower() == 'open_unmix':
            mean = np.load(os.path.join('..', 'Models', 'mean_db.npy'))
            std = np.load(os.path.join('..', 'Models', 'std_db.npy'))

            mean = np.expand_dims(mean, axis=-1)
            std = np.expand_dims(std, axis=-1)

            model = create_model_open_unmix_1_source(input_shape=input_shape, optimizer=optimizer, loss=loss,
                                                     n_fft=n_fft, max_freq=max_freq, sr=sr, batch_size=batch_size,
                                                     mean=mean, std=std)
        elif network.lower() == 'u_net':
            model = create_model_u_net_1_source(input_shape=input_shape, optimizer=optimizer, loss=loss,
                                                initial_filters=initial_filters, stride=stride, kernel_size=kernel_size,
                                                drop_out=drop_out)
            # model = create_model_u_net_1_source_maxpooling(input_shape=input_shape, optimizer=optimizer, loss=loss,
            #                                     initial_filters=initial_filters, stride=stride, kernel_size=kernel_size,
            #                                     drop_out=drop_out)
        elif network.lower() == 'cdae':
            model = create_model_cdae_1_source(input_shape=input_shape, optimizer=optimizer, loss=loss)

        else:
            raise Exception('Network type is not correct!')

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1,
              callbacks=callbacks)
