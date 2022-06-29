from preprocess import preprocess_1_source, preprocess_all_sources, preprocess_inverse, preprocess_1_source_aug, preprocess_1_source_aug_single_instrument
from train import training
from prediction import predict
from imports import *
from utils import get_name
from custom_callbacks import *
from lr_schedulers import *
from resume_training import resume_training


def main():
    preproc = False  # True for preprocessing
    train = True  # True for training
    pred = False  # True for predicting

    ######### PREPROCESSING PARAMETERS #########
    multiple_sources = False  # True for separating every source at the same time, False for one source
    dataset = 'MUSDB18'

    resample = True  # True for down-sampling every song
    sr = 8192  # Hertz
    window_length = 12  # Seconds
    overlap = 75  # Percent
    window_type = 'rect'

    compute_spect = True  # False for using waveforms as inputs to the NN, True for using spectrograms
    dB = True  # True for using spectrogram in dB
    n_fft = 1024  # Frame size for spectrograms
    hop_length = 768  # Hop length in samples for spectrograms

    extra_song_shuffle = True  # Applies shuffling between songs
    intra_song_shuffle = False  # Applies shuffling between windows of a song

    normalize = False  # Brings the data in a [-1, 1] range
    normalize01 = True  # Brings the data in a [0, 1] range
    standardize = False  # Brings the data in a normal distribution of mean = 0 and std = 1
    normalize_from_dataset = False  # True if the normalization uses statistics of the whole dataset

    source = 'vocals'  # Source to separate, for MUSDB18 can be 'bass', 'drums', 'vocals', 'other'

    aug = True
    augments = ['gaussian_noise', 'gain', 'reverb', 'overdrive', 'pitch']

    # The name format is:
    # signal-rep_dataset_sr_sr-value_window_win-len-value_overlap_percent_win-type_nfft_nfft-value_hop_hop-len-value_shuffle-mode_norm-mode_source'
    tfrecord_path = os.path.join('..', 'TFRecords')
    if preproc:
        name = get_name(compute_spect=compute_spect, dataset=dataset, sr=sr, window_length=window_length,
                        overlap=overlap, window_type=window_type, dB=dB, n_fft=n_fft, hop_length=hop_length,
                        extra_shuffle=extra_song_shuffle, intra_shuffle=intra_song_shuffle,
                        normalize_from_dataset=normalize_from_dataset, normalize=normalize, normalize01=normalize01,
                        standardize=standardize, multiple_sources=multiple_sources, source=source, aug=aug)
        writer_train = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_train.tfrecord'))
        writer_val = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_val.tfrecord'))
        statistics_path = os.path.join('..', 'Cardinality', '_'.join(name.split('_')[1:13]) + '_statistics.pkl')
        card_txt = name + '.txt'

    ######### TRAINING PARAMETERS #########
    network = 'u_net'
    train_aug = True
    train_multiple_sources = False  # True for separating every source at the same time, False for one source
    train_window_length = 12
    train_overlap = 75
    train_window_type = 'rect'
    train_compute_spect = True
    train_dB = True
    train_n_fft = 1024
    train_hop_length = 768
    train_sr = 8192  # For U-net, the audio should be resampled to 8192 Hz
    if train_compute_spect:
        if network == 'open_unmix':
            input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)))
        elif network == 'u_net':
            input_shape = (train_n_fft // 2, int(np.ceil((train_window_length * train_sr) / train_hop_length)), 1)
        elif network == 'cdae':
            input_shape = (train_n_fft // 2 + 1, int(np.ceil((train_window_length * train_sr) / train_hop_length)), 1)
    else:
        input_shape = (int(train_window_length * train_sr),)

    max_freq = 16000  # For Open-Unmix
    initial_filters = 16  # For U-net
    stride = (2, 2)  # For U-net
    kernel_size = (5, 5)  # For U-net
    train_batch_size = 128
    shuffle_buffer_size = 16 * train_batch_size
    epochs = 10000
    drop_out = 0.5
    early_stop_patience = 14
    learning_rate = 0.001
    lr_decay_patience = 100
    lr_decay_gamma = 0.1
    train_name = get_name(compute_spect=train_compute_spect, dataset=dataset, sr=train_sr,
                          window_length=train_window_length, overlap=train_overlap, window_type=train_window_type,
                          dB=train_dB, n_fft=train_n_fft, hop_length=train_hop_length, extra_shuffle=True,
                          intra_shuffle=False, normalize_from_dataset=False, normalize=False, normalize01=True,
                          standardize=False, multiple_sources=False, source=source, aug=train_aug)

    train_path = os.path.join(tfrecord_path, train_name + '_train.tfrecord')
    val_path = os.path.join(tfrecord_path, train_name + '_val.tfrecord')
    save_model_name = 'u_net_17_vocals_augx.h5'
    best_save_model_name = save_model_name.split('.')[0] + '_best.h5'
    model_path = os.path.join('..', 'Models')
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    log_name = '../Logs/log_' + save_model_name.split('.')[0] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    initial_epoch = 0
    f = open(os.path.join('..', 'Cardinality', ''.join(save_model_name.split('.h5')[:-1]) + '_epoch.txt'), 'w+')
    f.write(str(initial_epoch))
    f.close()

    best_index = 2
    callbacks = [
        TensorBoard(log_dir=log_name),
        ModelCheckpoint(os.path.join(model_path, save_model_name), monitor='val_loss', verbose=1, save_best_only=False),
        ModelCheckpoint(os.path.join(model_path, best_save_model_name), monitor='val_loss', verbose=1,
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=lr_decay_patience, factor=lr_decay_gamma, min_delta=100, verbose=1),
        Spectrogram_Callback_1_source(sr=train_sr, hop_length=train_hop_length, val_tfrecord_path=val_path,
                                      batch_size=train_batch_size, model_name=save_model_name.split('.h5')[0]),
        #LearningRateScheduler(reset_scheduler, verbose=1)
        GetEpoch(''.join(save_model_name.split('.h5')[:-1]))
    ]

    ######### PREDICTION PARAMETERS #########
    pred_source = 'other'
    pred_aug = False
    pred_multiple_sources = False
    pred_resample = True
    pred_sr = 8192
    pred_window_length = 12
    pred_overlap = 75
    pred_window_type = 'rect'

    pred_compute_spect = True
    pred_dB = True
    pred_n_fft = 1024
    pred_hop_length = 768

    pred_normalize = False
    pred_normalize01 = True
    pred_standardize = False
    pred_normalize_from_dataset = False

    test_path = os.path.join('..', 'Datasets', 'Test')
    test_song = 'test2.wav'  # one of train_1.wav, val_10.wav or test.wav
    load_model_name = 'u_net_17_other.h5'
    model_path = os.path.join('..', 'Models')
    save_song_path = os.path.join('..', 'Predictions', 'Wiener')
    pred_batch_size = 16
    pred_name = get_name(compute_spect=pred_compute_spect, dataset=dataset, sr=pred_sr,
                         window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
                         dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, extra_shuffle=True,
                         intra_shuffle=False, normalize_from_dataset=pred_normalize_from_dataset,
                         normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
                         multiple_sources=pred_multiple_sources, source=pred_source, aug=pred_aug)
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
            if not aug:
                preprocess_1_source(dataset=dataset, resample=resample, sr=sr, window_length=window_length, overlap=overlap,
                                window_type=window_type, compute_spect=compute_spect, dB=dB, n_fft=n_fft,
                                hop_length=hop_length, writer_train=writer_train, writer_val=writer_val, source=source,
                                extra_song_shuffle=extra_song_shuffle, intra_song_shuffle=intra_song_shuffle,
                                normalize_from_dataset=normalize_from_dataset, statistics_path=statistics_path,
                                normalize=normalize, normalize01=normalize01, standardize=standardize,
                                card_txt=card_txt, network=network)
            else:
                preprocess_1_source_aug_single_instrument(dataset=dataset, resample=resample, sr=sr, window_length=window_length,
                                    overlap=overlap,
                                    window_type=window_type, compute_spect=compute_spect, dB=dB, n_fft=n_fft,
                                    hop_length=hop_length, writer_train=writer_train, writer_val=writer_val,
                                    source=source,
                                    extra_song_shuffle=extra_song_shuffle, intra_song_shuffle=intra_song_shuffle,
                                    normalize_from_dataset=normalize_from_dataset, statistics_path=statistics_path,
                                    normalize=normalize, normalize01=normalize01, standardize=standardize,
                                    card_txt=card_txt, network=network, augments=augments)

    if train:
        try:
            training(network=network, train_path=train_path, val_path=val_path, batch_size=train_batch_size,
                 shuffle_buffer_size=shuffle_buffer_size, input_shape=input_shape, loss=loss, optimizer=optimizer,
                 drop_out=drop_out, epochs=epochs, callbacks=callbacks, multiple_sources=train_multiple_sources,
                 n_fft=train_n_fft, max_freq=max_freq, sr=train_sr, initial_filters=initial_filters, stride=stride,
                 kernel_size=kernel_size)
        except:
            with open(os.path.join('..', 'Cardinality', ''.join(save_model_name.split('.h5')[:-1]) + '_epoch.txt'), 'r') as f:
                initial_epoch = int(f.read())
            while initial_epoch < epochs:
                try:
                    best_save_model_name = save_model_name.split('.')[0] + '_best' + str(best_index) + '.h5'
                    callbacks[2] = ModelCheckpoint(os.path.join(model_path, best_save_model_name), monitor='val_loss',
                                                   verbose=1, save_best_only=True)

                    resume_training(network=network, train_path=train_path, val_path=val_path, model_path=model_path,
                                    save_model_name=save_model_name, train_batch_size=train_batch_size,
                                    shuffle_buffer_size=shuffle_buffer_size, epochs=epochs, initial_epoch=initial_epoch,
                                    callbacks=callbacks)
                except:
                    with open(os.path.join('..', 'Cardinality', ''.join(save_model_name.split('.h5')[:-1]) + '_epoch.txt'), 'r') as f:
                        initial_epoch = int(f.read())
                        best_index += 1


    if pred:
        predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
                multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
                sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
                dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
                normalize_from_dataset=pred_normalize_from_dataset, statistics_path=pred_statistics_path,
                normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
                batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song), network=network)


if __name__ == '__main__':
    main()
