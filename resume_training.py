from Py.custom_losses import L11_norm, L2_norm
from utils import *
from custom_callbacks import *
from custom_layers import *


######### TRAINING PARAMETERS #########
# network = 'u_net'
# train_multiple_sources = False  # True for separating every source at the same time, False for one source
# train_window_length = 12
# train_overlap = 90
# train_window_type = 'rect'
# train_compute_spect = True
# train_dB = True
# train_n_fft = 1024
# train_hop_length = 768
# train_sr = 8192  # For U-net, the audio should be resampled to 8192 Hz
# if train_compute_spect:
#     if network == 'open_unmix':
#         input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)))
#     elif network == 'u_net':
#         input_shape = (train_n_fft // 2, int(np.ceil((train_window_length * train_sr) / train_hop_length)), 1)
#     elif network == 'cdae':
#         input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)), 1)
# else:
#     input_shape = (int(train_window_length * train_sr),)
# max_freq = 16000  # For Open-Unmix
# initial_filters = 16  # For U-net
# stride = (2, 2)  # For U-net
# kernel_size = (5, 5)  # For U-net
# train_batch_size = 16
# shuffle_buffer_size = 128 * train_batch_size
# epochs = 1000
# drop_out = 0.5
# early_stop_patience = 14
# learning_rate = 0.001
# lr_decay_patience = 20
# lr_decay_gamma = 0.3
# train_name = get_name(compute_spect=train_compute_spect, dataset='MUSDB18', sr=train_sr,
#                       window_length=train_window_length, overlap=train_overlap, window_type=train_window_type,
#                       dB=train_dB, n_fft=train_n_fft, hop_length=train_hop_length, extra_shuffle=True,
#                       intra_shuffle=False, normalize_from_dataset=False, normalize=False, normalize01=True,
#                       standardize=False, multiple_sources=False, source='vocals')
# tfrecord_path = os.path.join('..', 'TFRecords')
# train_path = os.path.join(tfrecord_path, train_name + '_train.tfrecord')
# val_path = os.path.join(tfrecord_path, train_name + '_val.tfrecord')
# save_model_name = 'u_net_17_vocals_ov_90.h5'
# best_save_model_name = 'u_net_17_vocals_ov_90_best3.h5'
# model_path = os.path.join('..', 'Models')
# loss = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# callbacks = [
#     TensorBoard(log_dir='../Logs/log_' + save_model_name.split('.')[0] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
#     ModelCheckpoint(os.path.join(model_path, save_model_name), monitor='val_loss', verbose=1, save_best_only=False),
#     ModelCheckpoint(os.path.join(model_path, best_save_model_name), monitor='val_loss', verbose=1, save_best_only=True),
#     # EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=0.01, verbose=1),
#     # ReduceLROnPlateau(monitor='val_loss', patience=lr_decay_patience, factor=lr_decay_gamma),
#     Spectrogram_Callback_1_source(sr=train_sr, hop_length=train_hop_length, val_tfrecord_path=val_path,
#                                   batch_size=train_batch_size, model_name=save_model_name.split('.h5')[0])
# ]


#########################################################################
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