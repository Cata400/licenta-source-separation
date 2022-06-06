import norbert
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
from scipy.io import wavfile

sr = 10000
t = np.arange(0, 12, 1/sr)
s1 = np.sin(2 * np.pi * 500 * t)
s2 = 2 * np.sin(2 * np.pi * 1000 * t)
s3 = 3 * np.sin(2 * np.pi * 2000 * t)
s4 = 4 * np.sin(2 * np.pi * 4000 * t)
s = s1 + s2 + s3 + s4

for i, si in enumerate([s1, s2, s3, s4]):
    plt.figure(), plt.plot(si[:(i + 1) * 25]), plt.show()

plt.figure(), plt.plot(s), plt.show()

s1z = s1 + np.random.normal(0, 0.01, s1.shape)
s2z = s2 + np.random.normal(0, 0.01, s1.shape)
s3z = s3 + np.random.normal(0, 0.01, s1.shape)
s4z = s4 + np.random.normal(0, 0.01, s1.shape)

plt.figure(), plt.plot(s1z + s2z + s3z + s4z), plt.show()
#
# for si in [s1z, s2z, s3z, s4z]:
#     plt.figure(), plt.plot(si), plt.show()

# S4 = np.abs(scipy.fft.fft(s4, 2**20))
# plt.figure(), plt.plot(S4), plt.show()

N_fft = 8192
hop = 2048 * 3

S = librosa.stft(s, n_fft=N_fft, hop_length=hop)
S1z = librosa.stft(s1z, n_fft=N_fft, hop_length=hop)
S2z = librosa.stft(s2z, n_fft=N_fft, hop_length=hop)
S3z = librosa.stft(s3z, n_fft=N_fft, hop_length=hop)
S4z = librosa.stft(s4z, n_fft=N_fft, hop_length=hop)

import librosa.display
img = librosa.display.specshow(np.abs(librosa.stft(s4, n_fft=N_fft, hop_length=hop)), y_axis='log', x_axis='time', sr=sr)
plt.colorbar(img)
plt.show()


St = S.transpose()
S1zt = S1z.transpose()
S2zt = S2z.transpose()
S3zt = S3z.transpose()
S4zt = S4z.transpose()

St = np.expand_dims(St, axis=-1)
S1zt = np.expand_dims(S1zt, axis=(-2, -1))
S2zt = np.expand_dims(S2zt, axis=(-2, -1))
S3zt = np.expand_dims(S3zt, axis=(-2, -1))
S4zt = np.expand_dims(S4zt, axis=(-2, -1))

S_concat = np.concatenate([S1zt, S2zt, S3zt, S4zt], axis=-1)
filtered_sources = norbert.wiener(S_concat, St, iterations=2, use_softmask=False)


S1zf = filtered_sources[:, :, 0, 0].transpose()
S2zf = filtered_sources[:, :, 0, 1].transpose()
S3zf = filtered_sources[:, :, 0, 2].transpose()
S4zf = filtered_sources[:, :, 0, 3].transpose()

s1f = librosa.istft(S1zf, hop_length=hop, win_length=N_fft)
s2f = librosa.istft(S2zf, hop_length=hop, win_length=N_fft)
s3f = librosa.istft(S3zf, hop_length=hop, win_length=N_fft)
s4f = librosa.istft(S4zf, hop_length=hop, win_length=N_fft)

for i, (si, siz, sif) in enumerate(zip([s1, s2, s3, s4], [s1z, s2z, s3z, s4z], [s1f, s2f, s3f, s4f])):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(si)
    ax2.plot(siz)
    ax3.plot(sif)

    ax1.set_title('Original')
    ax2.set_title('Prezis')
    ax3.set_title('Filtrat')

    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    N = 1024
    Si = np.abs(np.fft.fft(si, N))
    Siz = np.abs(np.fft.fft(siz, N))
    Sif = np.abs(np.fft.fft(sif, N))

    freqs = np.fft.rfftfreq(N, d=1/sr)


    ax1.plot(freqs, Si[:N // 2 + 1])
    ax2.plot(freqs, Siz[:N // 2 + 1])
    ax3.plot(freqs, Sif[:N // 2 + 1])

    print(np.max(Si), np.max(Siz), np.max(Sif))
    print(freqs.shape)

    # ax1.plot(freqs[:len(freqs) // 5], Si[:len(freqs) // 5])
    # ax2.plot(freqs[:len(freqs) // 5], Siz[:len(freqs) // 5])
    # ax3.plot(freqs[:len(freqs) // 5], Sif[:len(freqs) // 5])

    ax1.set_title('Original')
    ax2.set_title('Prezis')
    ax3.set_title('Filtrat')

    plt.show()

    wavfile.write('sin_' + str(i) + '.wav', sr, si)
    wavfile.write('sin_' + str(i) + '_filtred.wav', sr, sif)


