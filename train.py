from utils import *
from models import *


def training(network, train_path, val_path, batch_size, shuffle_buffer_size, input_shape, loss, optimizer,
          drop_out, epochs, callbacks, multiple_sources, n_fft, max_freq, sr):

    train_dataset = tf.data.TFRecordDataset(train_path)
    val_dataset = tf.data.TFRecordDataset(val_path)

    if multiple_sources:
        train_dataset = train_dataset.map(parse_and_decode_all_sources)
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.map(parse_and_decode_all_sources)
        val_dataset = val_dataset.cache()
    else:
        train_dataset = train_dataset.map(parse_and_decode_1_source)
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.map(parse_and_decode_1_source)
        val_dataset = val_dataset.cache()

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

    if not multiple_sources:
        if network == 'open_unmix':
            model = create_model_open_unmix_1_source(input_shape=input_shape, optimizer=optimizer, loss=loss,
                                                     n_fft=n_fft, max_freq=max_freq, sr=sr)
        elif network == 'u_net':
            model = create_model_u_net_1_source(input_shape=input_shape, optimizer=optimizer, loss=loss,
                                                drop_out=drop_out)
        else:
            raise Exception('Network type is not correct!')

    model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset, verbose=1,
              callbacks=callbacks)
