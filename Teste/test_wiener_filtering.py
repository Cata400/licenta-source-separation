import numpy as np
import librosa
import os
from utils import sigwin

sr = 8192  # Hertz
window_length = 12  # Seconds
overlap = 75  # Percent
window_type = 'rect'

n_fft = 1024  # Frame size for spectrograms
hop_length = 768  # Hop length in samples for spectrograms

x, _ = librosa.load(os.path.join('..', '..', 'Predictions', 'Wiener', 'train_1_u_net_17_vocals_vocals.wav'), sr=None, mono=True)
x_filt, _ = librosa.load(os.path.join('..', '..', 'Predictions', 'Wiener', 'Filtered', 'filtered_train_1_u_net_17__vocals.wav'), sr=None, mono=True)

x_win = sigwin(x, window_length * sr, window_type, overlap)
x_filt_win = sigwin(x_filt, window_length * sr, window_type, overlap)

for window_index in range(x_win.shape[0]):
    x_spect = np.abs(librosa.stft(x_win[window_index], n_fft=n_fft, hop_length=hop_length), dtype='float32')
    x_filt_spect = np.abs(librosa.stft(x_filt_win[window_index], n_fft=n_fft, hop_length=hop_length), dtype='float32')

    print(np.allclose(x_spect, x_filt_spect))