import librosa
import numpy as np
from scipy.io import wavfile

path = '../../Datasets/Test/R.E.M. - Losing My Religion Lyrics.wav'
path2 = '../../Datasets/Test/R.E.M. - Losing My Religion Lyrics 44100.wav'

test, sr = librosa.load(path, sr=44100)
wavfile.write(path2, 44100, test)
