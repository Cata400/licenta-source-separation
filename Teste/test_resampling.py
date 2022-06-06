import librosa
from scipy.io import wavfile
import os
from sklearn.metrics import mean_squared_error
import numpy as np

x1, sr1 = librosa.load(os.path.join('..', '..', 'Datasets', 'Test', 'test2.wav'), sr=None, mono=True)
print(sr1)

# x2 = librosa.resample(x1, orig_sr=sr1, target_sr=sr2)
# x3 = librosa.resample(x2, orig_sr=sr2, target_sr=sr1)
#
# wavfile.write('train_1_vocals_8k.wav', sr2, x2)
# wavfile.write('train_1_vocals_44k.wav', sr1, x3)
# wavfile.write('ceva.wav', sr1, x2)
#
# min_length = np.min([len(x1), len(x3)])
# print(mean_squared_error(x1[:min_length], x3[:min_length]))