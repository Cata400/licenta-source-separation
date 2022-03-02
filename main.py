from preprocess import preprocess_1_source, preprocess_all_sources
from train import training
from prediction import predict
from imports import *
from utils import get_name
from custom_callbacks import *


def main():
    preproc = False  # True for preprocessing
    train = False  # True for training
    pred = True  # True for predicting

    ######### PREPROCESSING PARAMETERS #########
    multiple_sources = False  # True for separating every source at the same time, False for one source
    dataset = 'MUSDB18'

    resample = False  # True for downsampling every song
    sr = 44100  # Hertz
    window_length = 6  # Seconds
    overlap = 75  # Percent
    window_type = 'rect'

    compute_spect = True  # False for using waveforms as inputs to the NN, True for using spectrograms
    dB = False           # True for using spectrogram in dB
    n_fft = 4096  # Frame size for spectrograms
    hop_length = 1024  # Hop length in samples for spectrograms

    extra_song_shuffle = True  # Applies shuffling between songs
    intra_song_shuffle = False  # Applies shuffling between windows of a song

    normalize = False  # Brings the data in a [-1, 1] range
    normalize01 = False  # Brings the data in a [0, 1] range
    standardize = False  # Brings the data in a normal distribution of mean = 0 and std = 1
    normalize_from_dataset = False  # True if the normalization uses statistics of the whole dataset

    source = 'vocals'  # Source to separate, for MUSDB18 can be 'bass', 'drums', 'vocals', 'other'

    # The name format is:
    # signal-rep_dataset_sr_sr-value_window_win-len-value_overlap_percent_win-type_nfft_nfft-value_hop_hop-len-value_shuffle-mode_norm-mode_source'
    tfrecord_path = os.path.join('..', 'TFRecords')
    if preproc:
        name = get_name(compute_spect=compute_spect, dataset=dataset, sr=sr, window_length=window_length,
                        overlap=overlap, window_type=window_type, dB=dB, n_fft=n_fft, hop_length=hop_length,
                        extra_shuffle=extra_song_shuffle, intra_shuffle=intra_song_shuffle,
                        normalize_from_dataset=normalize_from_dataset, normalize=normalize, normalize01=normalize01,
                        standardize=standardize, multiple_sources=multiple_sources, source=source)
        writer_train = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_train.tfrecord'))
        writer_val = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_val.tfrecord'))
        statistics_path = os.path.join('..', 'Cardinality', '_'.join(name.split('_')[1:13]) + '_statistics.pkl')
        card_txt = name + '.txt'


    ######### TRAINING PARAMETERS #########
    train_multiple_sources = False  # True for separating every source at the same time, False for one source
    train_window_length = 6
    train_overlap = 75
    train_window_type = 'rect'
    train_compute_spect = True
    train_dB = False
    train_n_fft = 4096
    train_hop_length = 1024
    train_sr = 44100        # For U-net, the audio should be resampled to 8192 Hz
    if train_compute_spect:
        input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)))
    else:
        input_shape = (int(train_window_length * train_sr),)
    max_freq = 16000        # For Open-Unmix
    initial_filters = 16    # For U-net
    stride = 2              # For U-net
    kernel_size = (5, 5)    # For U-net
    train_batch_size = 16
    shuffle_buffer_size = 16 * train_batch_size
    epochs = 1000
    drop_out = 0.5
    early_stop_patience = 14
    learning_rate = 0.001
    lr_decay_patience = 20
    lr_decay_gamma = 0.3
    network = 'open_unmix'
    train_name = get_name(compute_spect=train_compute_spect, dataset=dataset, sr=train_sr,
                          window_length=train_window_length, overlap=train_overlap, window_type=train_window_type,
                          dB=train_dB, n_fft=train_n_fft, hop_length=train_hop_length, extra_shuffle=True,
                          intra_shuffle=False, normalize_from_dataset=False, normalize=False, normalize01=False,
                          standardize=False, multiple_sources=False, source=source)
    train_path = os.path.join(tfrecord_path, train_name + '_train.tfrecord')
    val_path = os.path.join(tfrecord_path, train_name + '_val.tfrecord')
    save_model_name = 'open_unmix_vocals2.h5'
    best_save_model_name = 'open_unmix_vocals_best2.h5'
    model_path = os.path.join('..', 'Models')
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks = [
        TensorBoard(log_dir='../Logs/log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        ModelCheckpoint(os.path.join(model_path, save_model_name), monitor='val_loss', verbose=1, save_best_only=False),
        ModelCheckpoint(os.path.join(model_path, best_save_model_name), monitor='val_loss', verbose=1,
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=lr_decay_patience, factor=lr_decay_gamma),
        Spectrogram_Callback_1_source(sr=train_sr, hop_length=train_hop_length, val_tfrecord_path=val_path,
                                      batch_size=train_batch_size, model_name=save_model_name.split('.h5')[0])
    ]


    ######### PREDICTION PARAMETERS #########
    pred_source = 'vocals'
    pred_multiple_sources = False
    pred_resample = False
    pred_sr = 44100
    pred_window_length = 6
    pred_overlap = 75
    pred_window_type = 'rect'

    pred_compute_spect = True
    pred_dB = False
    pred_n_fft = 4096
    pred_hop_length = 1024

    pred_normalize = False
    pred_normalize01 = False
    pred_standardize = False
    pred_normalize_from_dataset = False

    test_path = os.path.join('..', 'Datasets', 'Test')
    test_song = 'R.E.M. - Losing My Religion Lyrics 44100.wav'
    load_model_name = 'open_unmix_vocals_8_best.h5'
    model_path = os.path.join('..', 'Models')
    save_song_path = os.path.join('..', 'Predictions')
    pred_batch_size = 16
    pred_name = get_name(compute_spect=pred_compute_spect, dataset=dataset, sr=pred_sr,
                         window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
                         dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, extra_shuffle=True,
                         intra_shuffle=False, normalize_from_dataset=pred_normalize_from_dataset,
                         normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
                         multiple_sources=pred_multiple_sources, source=pred_source)
    pred_statistics_path = os.path.join('..', 'Cardinality', '_'.join(pred_name.split('_')[1:13]) + '_statistics.pkl')

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        exit()

    if preproc:
        if multiple_sources:
            preprocess_all_sources(dataset=dataset, resample=resample, sr=sr, window_length=window_length,
                                   overlap=overlap, window_type=window_type, compute_spect=compute_spect, dB=dB,
                                   n_fft=n_fft, hop_length=hop_length, writer_train=writer_train, writer_val=writer_val,
                                   extra_song_shuffle=extra_song_shuffle, intra_song_shuffle=intra_song_shuffle,
                                   normalize_from_dataset=normalize_from_dataset, statistics_path=statistics_path,
                                   normalize=normalize, normalize01=normalize01, standardize=standardize,
                                   card_txt=card_txt)
        else:
            preprocess_1_source(dataset=dataset, resample=resample, sr=sr, window_length=window_length, overlap=overlap,
                                window_type=window_type, compute_spect=compute_spect, dB=dB, n_fft=n_fft,
                                hop_length=hop_length, writer_train=writer_train, writer_val=writer_val, source=source,
                                extra_song_shuffle=extra_song_shuffle, intra_song_shuffle=intra_song_shuffle,
                                normalize_from_dataset=normalize_from_dataset, statistics_path=statistics_path,
                                normalize=normalize, normalize01=normalize01, standardize=standardize,
                                card_txt=card_txt)

    if train:
        training(network=network, train_path=train_path, val_path=val_path, batch_size=train_batch_size,
                 shuffle_buffer_size=shuffle_buffer_size, input_shape=input_shape, loss=loss, optimizer=optimizer,
                 drop_out=drop_out, epochs=epochs, callbacks=callbacks, multiple_sources=train_multiple_sources,
                 n_fft=train_n_fft, max_freq=max_freq, sr=train_sr, initial_filters=initial_filters, stride=stride,
                 kernel_size=kernel_size)

    if pred:
        predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
                multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
                sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
                dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
                normalize_from_dataset=pred_normalize_from_dataset, statistics_path=pred_statistics_path,
                normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
                batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song))


if __name__ == '__main__':
    main()
