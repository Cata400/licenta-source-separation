import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import sigwin
import os


# filename = librosa.ex('trumpet')
filename = os.path.join('..', '..', 'Datasets', 'Test', 'train_1.wav')
x, sr = librosa.load(filename, sr=44100, mono=True)
# print(len(x) / sr)

sr = 22050  # Hertz
window_length = 1  # Seconds
overlap = 25  # Percent
window_type = 'rect'

n_fft = 2048  # Frame size for spectrograms
hop_length = 1536  # Hop length in samples for spectrograms

x_win = sigwin(x, window_length * sr, window_type, overlap)

for y in x_win:
    X = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    print(X.shape)
    break
