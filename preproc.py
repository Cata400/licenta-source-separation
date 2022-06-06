import numpy as np
import tensorflow as tf
import os
import librosa
from utils2 import *

################ PARAMETERS ####################################
dataset = 'MUSDB18_15_mel'

resample = True  # True for downsampling every song
sr = 44100  # Hertz
window_length = 6  # Seconds
overlap = 75  # Percent
window_type = 'rect'
n_fft = 4096  # Frame size for spectrograms
hop_length = 1024  # Hop length in samples for spectrograms

extra_song_shuffle = True  # Applies shuffling between songs

source = 'vocals'  # Source to separate, for MUSDB18 can be 'bass', 'drums', 'vocals', 'other'

tfrecord_path = os.path.join('TFRecords')
name = 'incercare'
writer_train = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_train.tfrecord'))
writer_val = tf.io.TFRecordWriter(os.path.join(tfrecord_path, name + '_val.tfrecord'))

#################### PREPROCESSING ############################
np.random.seed(42)

for subfolder in sorted(os.listdir(os.path.join('Datasets', dataset))):
    path = os.path.join('Datasets', dataset, subfolder)
    songs = os.listdir(path)

    if extra_song_shuffle:
        np.random.shuffle(songs)
    else:
        songs = sorted(songs)

    for i, song in enumerate(songs):
        # if i == 2:
        #     break
        print(i, song)
        if resample:
            x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
            y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)
        else:
            sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
            x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
            y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)

        x_windows = sigwin(x, window_length * sr, window_type, overlap)
        y_windows = sigwin(y, window_length * sr, window_type, overlap)

        for window_index in range(x_windows.shape[0]):
            x_window_spect = np.abs(librosa.stft(x_windows[window_index], n_fft=n_fft,
                                                 hop_length=hop_length), dtype='float32')
            y_window_spect = np.abs(librosa.stft(y_windows[window_index], n_fft=n_fft,
                                                 hop_length=hop_length), dtype='float32')

            if subfolder == 'train':
                train_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                        y_window_spect.astype(np.float32))
                writer_train.write(train_serialized_spectrograms)
            elif subfolder == 'val':
                val_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                      y_window_spect.astype(np.float32))
                writer_val.write(val_serialized_spectrograms)
