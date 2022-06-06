from scipy.io import wavfile
from scipy.signal import spectrogram, stft
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from utils import sigwin
import librosa.display
from sklearn.metrics import mean_squared_error

path = os.path.join('..', '..', 'Datasets', 'MUSDB18', 'train', 'A Classic Education - NightOwl', 'mixture.wav')

fs, x = wavfile.read(path)
x = np.mean(x, axis=1)
x = x / np.max(np.abs(x))
y, sr = librosa.load(path, sr=None, mono=True)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x)
ax2.plot(y)
plt.show()

x_windows = sigwin(x, 6 * fs, 'rect', 75)
y_windows = sigwin(y, 6 * sr, 'rect', 75)

for window_index in range(x_windows.shape[0]):
    if window_index == 0:
        print(mean_squared_error(x_windows[window_index], y_windows[window_index]))

    _, _, x_window_spect = stft(x_windows[window_index], fs=fs, window='hann', nperseg=4096,
                                       noverlap=4096 - 1024, nfft=4096)
    y_window_spect = librosa.stft(y_windows[window_index], n_fft=4096, hop_length=1024)

    x_window_spect = np.abs(x_window_spect, dtype='float32')
    y_window_spect = np.abs(y_window_spect, dtype='float32')

    break

print(x_window_spect.shape, y_window_spect.shape)
fig = plt.figure()
img1 = librosa.display.specshow(x_window_spect, y_axis='log', sr=44100, hop_length=1024, x_axis='time')
fig.colorbar(img1)
plt.show()

fig = plt.figure()
img2 = librosa.display.specshow(y_window_spect, y_axis='log', sr=44100, hop_length=1024, x_axis='time')
fig.colorbar(img2)
plt.show()
