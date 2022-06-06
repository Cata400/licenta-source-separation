from utils import sigwin
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt

x, sr = librosa.load(os.path.join('..', '..', 'Datasets', 'Test', 'train_1_vocals.wav'), sr=8192, mono=True)
x_g, _ = librosa.load(os.path.join('..', '..', 'Predictions', 'Wiener', 'train_1_u_net_17_vocals_vocals.wav'), sr=8192, mono=True)
x_w, _ = librosa.load(os.path.join('..', '..', 'Predictions', 'Wiener', 'Filtered_orig_phase', 'filtered_train_1_u_net_17__vocals.wav'), sr=8192, mono=True)

sr = 8192  # Hertz
window_length = 12  # Seconds
overlap = 75  # Percent
window_type = 'rect'

n_fft = 1024  # Frame size for spectrograms
hop_length = 768  # Hop length in samples for spectrograms

x_win = sigwin(x, window_length * sr, window_type, overlap)
x_g_win = sigwin(x_g, window_length * sr, window_type, overlap)
x_w_win = sigwin(x_w, window_length * sr, window_type, overlap)

window_index = 20

x_spect = np.abs(librosa.stft(x_win[window_index], n_fft=n_fft, hop_length=hop_length), dtype='float32')
x_g_spect = np.abs(librosa.stft(x_g_win[window_index], n_fft=n_fft, hop_length=hop_length), dtype='float32')
x_w_spect = np.abs(librosa.stft(x_w_win[window_index], n_fft=n_fft, hop_length=hop_length), dtype='float32')

x_spect = librosa.amplitude_to_db(x_spect, ref=np.max)
x_g_spect = librosa.amplitude_to_db(x_g_spect, ref=np.max)
x_w_spect = librosa.amplitude_to_db(x_w_spect, ref=np.max)

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
img1 = librosa.display.specshow(x_spect, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[0])
img2 = librosa.display.specshow(x_g_spect, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
img3 = librosa.display.specshow(x_w_spect, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[2])
fig.colorbar(img1, ax=ax)
plt.show()