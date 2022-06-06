import norbert
import librosa
import numpy as np
import os
from utils import sigwin, sigrec
from scipy.io import wavfile


song_name = 'train_1.wav'
best_model = False
model_name = 'u_net_17'
if best_model:
    model_name += '_best'
test_path = os.path.join('..', 'Datasets', 'Test', song_name)
source_path = os.path.join('..', 'Predictions', 'Wiener')
filtered_path = os.path.join('..', 'Predictions', 'Wiener', 'Filtered50')
file_name = song_name.split('.')[0] + '_' + model_name + '_'


sr = 8192  # Hertz
window_length = 12  # Seconds
overlap = 75  # Percent
window_type = 'rect'

n_fft = 1024  # Frame size for spectrograms
hop_length = 768  # Hop length in samples for spectrograms


mixture, sr = librosa.load(test_path, sr=8192, mono=True)
if best_model:
    vocals, _ = librosa.load(os.path.join(source_path, '_'.join(file_name.split('_')[:-2]) + '_vocals_best_vocals.wav'), mono=True, sr=None)
    bass, _ = librosa.load(os.path.join(source_path, '_'.join(file_name.split('_')[:-2]) + '_bass_best_bass.wav'), mono=True, sr=None)
    drums, _ = librosa.load(os.path.join(source_path, '_'.join(file_name.split('_')[:-2]) + '_drums_best_drums.wav'), mono=True, sr=None)
    other, _ = librosa.load(os.path.join(source_path, '_'.join(file_name.split('_')[:-2]) + '_other_best_other.wav'), mono=True, sr=None)
else:
    vocals, _ = librosa.load(os.path.join(source_path, file_name + 'vocals_vocals.wav'), mono=True, sr=None)
    bass, _ = librosa.load(os.path.join(source_path, file_name + 'bass_bass.wav'), mono=True, sr=None)
    drums, _ = librosa.load(os.path.join(source_path, file_name + 'drums_drums.wav'), mono=True, sr=None)
    other, _ = librosa.load(os.path.join(source_path, file_name + 'other_other.wav'), mono=True, sr=None)

print('Loading sources done!')

# mixture = vocals + bass

mixture = mixture[:len(vocals)]
# all = mixture - vocals

mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
other_windows = sigwin(other, window_length * sr, window_type, overlap)

# all_windows = sigwin(all, window_length * sr, window_type, overlap)

print('Windowing sources done!')

filtered_vocals = []
filtered_bass = []
filtered_drums = []
filtered_other = []

# filtered_all = []

for window_index in range(mixture_windows.shape[0]):
    mixture_window_stft = librosa.stft(mixture_windows[window_index], n_fft=n_fft, hop_length=hop_length)

    mixture_window_spect = np.abs(mixture_window_stft, dtype='float32')
    bass_window_spect = np.abs(librosa.stft(bass_windows[window_index], n_fft=n_fft,
                                            hop_length=hop_length), dtype='float32')
    drums_window_spect = np.abs(librosa.stft(drums_windows[window_index], n_fft=n_fft,
                                             hop_length=hop_length), dtype='float32')
    vocals_window_spect = np.abs(librosa.stft(vocals_windows[window_index], n_fft=n_fft,
                                              hop_length=hop_length), dtype='float32')
    other_window_spect = np.abs(librosa.stft(other_windows[window_index], n_fft=n_fft,
                                             hop_length=hop_length), dtype='float32')

    # all_window_spect = np.abs(librosa.stft(all_windows[window_index], n_fft=n_fft,
    #                                          hop_length=hop_length), dtype='float32')

    mixture_window_stft = mixture_window_stft.transpose()
    mixture_window_spect = mixture_window_spect.transpose()
    vocals_window_spect = vocals_window_spect.transpose()
    bass_window_spect = bass_window_spect.transpose()
    drums_window_spect = drums_window_spect.transpose()
    other_window_spect = other_window_spect.transpose()

    # all_window_spect = all_window_spect.transpose()

    mixture_window_stft = np.expand_dims(mixture_window_stft, axis=-1)
    vocals_window_spect = np.expand_dims(vocals_window_spect, axis=(-2, -1))
    bass_window_spect = np.expand_dims(bass_window_spect, axis=(-2, -1))
    drums_window_spect = np.expand_dims(drums_window_spect, axis=(-2, -1))
    other_window_spect = np.expand_dims(other_window_spect, axis=(-2, -1))

    # all_window_spect = np.expand_dims(all_window_spect, axis=(-2, -1))

    sources = np.concatenate((vocals_window_spect, bass_window_spect, drums_window_spect, other_window_spect), axis=-1)
    # sources = np.concatenate((vocals_window_spect, bass_window_spect), axis=-1)

    # sources = np.concatenate((vocals_window_spect, all_window_spect), axis=-1)
    filtered_sources = norbert.wiener(sources, mixture_window_stft, iterations=1, use_softmask=False)

    filtered_vocals_spect = filtered_sources[:, :, 0, 0].transpose()
    filtered_bass_spect = filtered_sources[:, :, 0, 1].transpose()
    filtered_drums_spect = filtered_sources[:, :, 0, 2].transpose()
    filtered_other_spect = filtered_sources[:, :, 0, 3].transpose()

    # filtered_all_spect = filtered_sources[:, :, 0, 1].transpose()

    filtered_vocals_window = librosa.istft(filtered_vocals_spect, hop_length=hop_length, win_length=n_fft)
    filtered_bass_window = librosa.istft(filtered_bass_spect, hop_length=hop_length, win_length=n_fft)
    filtered_drums_window = librosa.istft(filtered_drums_spect, hop_length=hop_length, win_length=n_fft)
    filtered_other_window = librosa.istft(filtered_other_spect, hop_length=hop_length, win_length=n_fft)

    # filtered_all_window = librosa.istft(filtered_all_spect, hop_length=hop_length, win_length=n_fft)

    filtered_vocals.append(filtered_vocals_window)
    filtered_bass.append(filtered_bass_window)
    filtered_drums.append(filtered_drums_window)
    filtered_other.append(filtered_other_window)

    # filtered_all.append(filtered_all_window)


print('Filtered windows done!')

new_vocals = sigrec(filtered_vocals, overlap, 'MEAN')
wavfile.write(os.path.join(filtered_path, 'filtered_' + file_name + '_vocals.wav'), sr, new_vocals)
print('Vocals filter done!')

new_bass = sigrec(filtered_bass, overlap, 'MEAN')
wavfile.write(os.path.join(filtered_path, 'filtered_' + file_name + '_bass.wav'), sr, new_bass)
print('Bass filter done!')

new_drums = sigrec(filtered_drums, overlap, 'MEAN')
wavfile.write(os.path.join(filtered_path, 'filtered_' + file_name + '_drums.wav'), sr, new_drums)
print('Drums filter done!')

new_other = sigrec(filtered_other, overlap, 'MEAN')
wavfile.write(os.path.join(filtered_path, 'filtered_' + file_name + '_other.wav'), sr, new_other)
print('Other filter done!')

# new_all = sigrec(filtered_all, overlap, 'MEAN')
# wavfile.write(os.path.join(filtered_path, 'filtered_' + file_name + '_all.wav'), sr, new_all)
# print('Vocals filter done!')