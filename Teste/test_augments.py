import librosa
import numpy as np
import os
from scipy.io import wavfile
from utils import data_aug

x, sr = librosa.load(os.path.join('..', '..', 'Datasets', 'Test', 'train_1.wav'), sr=None, mono=True)
y, sr = librosa.load(os.path.join('..', '..', 'Datasets', 'Test', 'train_1_vocals.wav'), sr=None, mono=True)

augments = ['noise', 'gain', 'reverb', 'phaser', 'overdrive', 'pitch']

for aug in augments:
    x_new, y_new = data_aug((x, y), aug)
    wavfile.write('train1_' + aug + '.wav', sr, x_new)
    wavfile.write('train1_vocals_' + aug + '.wav', sr, y_new)