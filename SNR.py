import numpy as np
import librosa
import os


def compute_SNR(orig, noisy):
    min_length = np.min([len(orig), len(noisy)])
    orig = orig[: min_length]
    noisy = noisy[: min_length]

    snr = 10 * np.log10(np.sum(orig ** 2) / np.sum((orig - noisy) ** 2))

    return snr


snr_path = filtered_path = os.path.join('..', 'Predictions', 'Wiener', 'Filtered')
orig_path = os.path.join('..', 'Datasets', 'Test')

train_44, _ = librosa.load(os.path.join(orig_path, 'train_1_vocals.wav'), sr=None, mono=True)
train_8, _ = librosa.load(os.path.join(orig_path, 'train_1_vocals.wav'), sr=8192, mono=True)

val_44, _ = librosa.load(os.path.join(orig_path, 'val_10_vocals.wav'), sr=None, mono=True)
val_8, _ = librosa.load(os.path.join(orig_path, 'val_10_vocals.wav'), sr=8192, mono=True)


for file in sorted(os.listdir(snr_path)):
    if 'vocals' in file:
        song_8, _ = librosa.load(os.path.join(snr_path, file), sr=None, mono=True)
        song_44 = librosa.resample(song_8, 8192, 44100)

        if 'test' not in file:
            snr44 = compute_SNR(train_44, song_44)
            snr8 = compute_SNR(train_8, song_8)

            print('{} SNR 44k = {} dB, SNR 8k = {} dB'.format(file, snr44, snr8))
