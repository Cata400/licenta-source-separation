from Py.custom_losses import L11_norm, L2_norm
from utils import *
from custom_callbacks import *
from custom_layers import *


def resume_training(network, train_path, val_path, model_path, save_model_name, train_batch_size, shuffle_buffer_size, epochs,
                    initial_epoch, callbacks):
    if network.lower() == 'open_unmix':
        model = tf.keras.models.load_model(os.path.join(model_path, save_model_name), custom_objects={'ScaleInLayer': ScaleInLayer,
                                                                       'ScaleOutLayer': ScaleOutLayer})
    elif network.lower() == 'u_net':
        try:
            model = tf.keras.models.load_model(os.path.join(model_path, save_model_name))
        except:
            model = tf.keras.models.load_model(os.path.join(model_path, save_model_name), custom_objects={'L11_norm': L11_norm})

    elif network.lower() == 'cdae':
        try:
            model = tf.keras.models.load_model(os.path.join(model_path, save_model_name))
        except:
            model = tf.keras.models.load_model(os.path.join(model_path, save_model_name), custom_objects={'L2_norm': L2_norm})

    model.summary()

    train_dataset = tf.data.TFRecordDataset(train_path)
    val_dataset = tf.data.TFRecordDataset(val_path)

    train_dataset = train_dataset.map(parse_and_decode_1_source)
    val_dataset = val_dataset.map(parse_and_decode_1_source)

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(train_batch_size)
    val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(train_batch_size)

    model.fit(train_dataset, batch_size=train_batch_size, epochs=epochs, validation_data=val_dataset, verbose=1,
              callbacks=callbacks, initial_epoch=initial_epoch)
