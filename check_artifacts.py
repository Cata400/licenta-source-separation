from utils import *


x, sr = librosa.load(os.path.join('..', 'Datasets', 'MUSDB18', 'train', 'A Classic Education - NightOwl', 'vocals.wav'), sr=8192, mono=True)
x_p, _ = librosa.load(os.path.join('..', 'Predictions', 'Wiener', 'train_1_u_net_17_vocals_vocals.wav'), sr=8192, mono=True)
x_w, _ = librosa.load(os.path.join('..', 'Predictions', 'Wiener', 'Filtered_orig_phase', 'filtered_train_1_u_net_17__vocals.wav'), sr=8192, mono=True)

resample = True  # True for down-sampling every song
sr = 8192  # Hertz
window_length = 0.5  # Seconds
overlap = 75  # Percent
window_type = 'hamming'

x_windows = sigwin(x, window_length * sr, window_type, overlap)
x_p_windows = sigwin(x_p, window_length * sr, window_type, overlap)
x_w_windows = sigwin(x_w, window_length * sr, window_type, overlap)

N = sr

for i, (win, win_p, win_w) in enumerate(zip(x_windows, x_p_windows, x_w_windows)):
    Win = np.abs(np.fft.fft(win, N))
    Win_p = np.abs(np.fft.fft(win_p, N))
    Win_w = np.abs(np.fft.fft(win_w, N))

    freqs = np.fft.rfftfreq(N, d=1/sr)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(freqs, Win[:N // 2 + 1])
    ax2.plot(freqs, Win_p[:N // 2 + 1])
    ax3.plot(freqs, Win_w[:N // 2 + 1])

    ax1.set_title('Original')
    ax2.set_title('Prezis')
    ax3.set_title('Filtrat')

    plt.show()
