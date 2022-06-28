import librosa
import numpy as np
import os
from scipy.io import wavfile

initial_path = os.path.join('..', 'Datasets', 'MUSDB18_predict_normed_Wiener')
normed_path = os.path.join('..', 'Datasets', 'MUSDB18_predict_normed_Wiener_normed')

for j, folder in enumerate(sorted(os.listdir(initial_path))):
    if not os.path.exists(os.path.join(normed_path, folder)):
        os.mkdir(os.path.join(normed_path, folder))
    for i, song in enumerate(sorted(os.listdir(os.path.join(initial_path, folder)))):
        print(i, song)
        if not os.path.exists(os.path.join(normed_path, folder, song)):
            os.mkdir(os.path.join(normed_path, folder, song))
        for file in sorted(os.listdir(os.path.join(initial_path, folder, song))):
            x, sr = librosa.load(os.path.join(initial_path, folder, song, file), mono=True, sr=None)
            x_normed = x / np.max(np.abs(x))
            wavfile.write(os.path.join(normed_path, folder, song, file), sr, x_normed)

