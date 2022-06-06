import numpy as np
import librosa
import os
from sklearn.metrics import mean_squared_error


path = '../../Datasets/MUSDB18/val'

for song in sorted(os.listdir(path)):
    mixture, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), mono=True)
    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), mono=True)
    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), mono=True)
    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), mono=True)
    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), mono=True)

    print(song, np.allclose(mixture, bass + drums + vocals + other), mean_squared_error(mixture, bass + drums + vocals + other))