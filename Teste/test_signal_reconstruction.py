from utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

mixture_path = "../../Datasets/MUSDB18/train/A Classic Education - NightOwl/mixture.wav"
source_path = "../../Datasets/MUSDB18/train/A Classic Education - NightOwl/vocals.wav"
sr = 8192
mixture, _ = librosa.load(mixture_path, sr=sr, mono=True)
source, _ = librosa.load(source_path, sr=sr, mono=True)
window_length = 12  # s
overlap = 75
window_type = 'rect'
n_fft = 1024
hop_length = 768

mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
source_windows = sigwin(source, window_length * sr, window_type, overlap)
print('windows done')
print('Initial windows shape: ', mixture_windows.shape)

mixture_phase = []
mixture_magnitude = []
source_phase = []
source_magnitude = []
mixture_maxim = []
mixture_minim = []
source_maxim = []
source_minim = []
ref = []
for mixture_window, source_window in zip(mixture_windows, source_windows):
    mixture_spect = librosa.stft(mixture_window, n_fft=n_fft, hop_length=hop_length)
    source_spect = librosa.stft(source_window, n_fft=n_fft, hop_length=hop_length)
    mixture_abs = np.abs(mixture_spect)
    source_abs = np.abs(source_spect)
    ref.append(np.max(mixture_abs))
    source_abs = librosa.amplitude_to_db(source_abs, ref=np.max(mixture_abs))
    mixture_abs = librosa.amplitude_to_db(mixture_abs, ref=np.max)
    np.clip(source_abs, -80, 0, out=source_abs)
    # print(np.max(source_abs), np.min(source_abs))
    mixture_maxim.append(np.max(mixture_abs))
    mixture_minim.append(np.min(mixture_abs))
    source_maxim.append(np.max(source_abs))
    source_minim.append(np.min(source_abs))
    # if np.max(mixture_abs) - np.min(mixture_abs) > 1:
    #     source_abs = (source_abs - np.min(mixture_abs)) / (np.max(mixture_abs) - np.min(mixture_abs))
    # else:
    #     source_abs = np.zeros(source_abs.shape)
    # if np.max(mixture_abs) - np.min(mixture_abs) > 1:
    #     mixture_abs = (mixture_abs - np.min(mixture_abs)) / (np.max(mixture_abs) - np.min(mixture_abs))
    # else:
    #     mixture_abs = np.zeros(mixture_abs.shape)
    mixture_abs = (mixture_abs + 80) / 80
    source_abs = (source_abs + 80) / 80
    mixture_magnitude.append(mixture_abs)
    source_magnitude.append(source_abs)

    mixture_ph = np.angle(mixture_spect)
    source_ph = np.angle(source_spect)
    mixture_phase.append(mixture_ph)
    source_phase.append(source_ph)

mixture_magnitude = np.asanyarray(mixture_magnitude)
source_magnitude = np.asanyarray(source_magnitude)
mixture_phase = np.asanyarray(mixture_phase)
source_phase = np.asanyarray(source_phase)
print('specs done')
print('Spectrogram list shape: ', mixture_magnitude.shape)

mixture_new_windows = []
source_new_windows = []
for mixture_abs, mixture_ph, source_abs, source_ph, mixture_max, mixture_min, source_max, source_min, ref_max in \
        zip(mixture_magnitude, mixture_phase, source_magnitude, source_phase, mixture_maxim, mixture_minim,
            source_maxim, source_minim, ref):
    mixture_abs = mixture_abs * 80 - 80
    source_abs = source_abs * 80 - 80
    # print(np.max(source_abs), np.min(source_abs))
    mixture_abs = librosa.db_to_amplitude(mixture_abs, ref=ref_max)
    source_abs = librosa.db_to_amplitude(source_abs, ref=ref_max)
    mixture_spect_new = np.multiply(mixture_abs, np.exp(1j * mixture_ph))
    source_spect_new = np.multiply(source_abs, np.exp(1j * mixture_ph))
    mixture_window_new = librosa.istft(mixture_spect_new, hop_length=hop_length, win_length=n_fft)
    source_window_new = librosa.istft(source_spect_new, hop_length=hop_length, win_length=n_fft)
    mixture_new_windows.append(mixture_window_new)
    source_new_windows.append(source_window_new)

print('new windows done')
mixture_new_windows = np.asanyarray(mixture_new_windows)
source_new_windows = np.asanyarray(source_new_windows)
print('New windows shape: ', mixture_new_windows.shape)
print('Initial song shape: ', mixture.shape)

mixture_new = sigrec(mixture_new_windows, overlap, 'MEAN')
source_new = sigrec(source_new_windows, overlap, 'MEAN')
print(np.asanyarray(mixture).shape, np.asanyarray(mixture_new).shape)
print(np.asanyarray(source).shape, np.asanyarray(source_new).shape)
wavfile.write('test_mixture.wav', sr, mixture_new)
wavfile.write('test_source.wav', sr, source_new)


mixture = mixture[:len(mixture_new)]
source = source[:len(source_new)]
print('Mixture MSE: ', mean_squared_error(mixture, mixture_new))
print('Mixture MAE: ', mean_absolute_error(mixture, mixture_new))
print('Source MSE: ', mean_squared_error(source, source_new))
print('Source MAE: ', mean_absolute_error(source, source_new))
