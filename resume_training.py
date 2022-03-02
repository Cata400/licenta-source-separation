from utils import *
from custom_callbacks import *
from custom_layers import *


######### TRAINING PARAMETERS #########
train_multiple_sources = False  # True for separating every source at the same time, False for one source
train_window_length = 6
train_overlap = 75
train_window_type = 'rect'
train_compute_spect = True
train_dB = False
train_n_fft = 4096
train_hop_length = 1024
train_sr = 44100  # For U-net, the audio should be resampled to 8192 Hz
if train_compute_spect:
    input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)))
else:
    input_shape = (int(train_window_length * train_sr),)
max_freq = 16000  # For Open-Unmix
initial_filters = 16  # For U-net
stride = 2  # For U-net
kernel_size = (5, 5)  # For U-net
train_batch_size = 16
shuffle_buffer_size = 16 * train_batch_size
epochs = 200
drop_out = 0.5
early_stop_patience = 14
learning_rate = 0.001
lr_decay_patience = 16
lr_decay_gamma = 0.3
network = 'open_unmix'
train_name = get_name(compute_spect=train_compute_spect, dataset='MUSDB18', sr=train_sr,
                      window_length=train_window_length, overlap=train_overlap, window_type=train_window_type,
                      dB=train_dB, n_fft=train_n_fft, hop_length=train_hop_length, extra_shuffle=True,
                      intra_shuffle=False, normalize_from_dataset=False, normalize=False, normalize01=False,
                      standardize=False, multiple_sources=False, source='vocals')
tfrecord_path = os.path.join('..', 'TFRecords')
train_path = os.path.join(tfrecord_path, train_name + '_train.tfrecord')
val_path = os.path.join(tfrecord_path, train_name + '_val.tfrecord')
save_model_name = 'open_unmix_vocals2.h5'
best_save_model_name = 'open_unmix_vocals2_best.h5'
model_path = os.path.join('..', 'Models')
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
callbacks = [
    TensorBoard(log_dir='../Logs/log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    ModelCheckpoint(os.path.join(model_path, save_model_name), monitor='val_loss', verbose=1, save_best_only=False),
    ModelCheckpoint(os.path.join(model_path, best_save_model_name), monitor='val_loss', verbose=1, save_best_only=True),
    # EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=0.01, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=lr_decay_patience, factor=lr_decay_gamma),
    Spectrogram_Callback_1_source(sr=train_sr, hop_length=train_hop_length, val_tfrecord_path=val_path,
                                  batch_size=train_batch_size, model_name=save_model_name.split('.h5')[0])
]


#########################################################################
model = tf.keras.models.load_model(os.path.join(model_path, 'open_unmix_vocals2.h5'),
                                   custom_objects={'ScaleInLayer': ScaleInLayer, 'ScaleOutLayer': ScaleOutLayer})
model.summary()

train_dataset = tf.data.TFRecordDataset(train_path)
val_dataset = tf.data.TFRecordDataset(val_path)

train_dataset = train_dataset.map(parse_and_decode_1_source)
val_dataset = val_dataset.map(parse_and_decode_1_source)

train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(train_batch_size)
val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(train_batch_size)

model.fit(train_dataset, batch_size=train_batch_size, epochs=epochs, validation_data=val_dataset, verbose=1,
          callbacks=callbacks, initial_epoch=60)