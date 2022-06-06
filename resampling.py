import librosa
from scipy.io import wavfile
import os
import numpy as np

x1, sr1 = librosa.load(os.path.join('..', 'Datasets', 'Test', 'test2.wav'), sr=None, mono=True)
sr2 = 44100

x2 = librosa.resample(x1, orig_sr=sr1, target_sr=sr2)

wavfile.write(os.path.join('..', 'Datasets', 'Test', 'test244.wav'), sr2, x2)

