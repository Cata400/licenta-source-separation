import norbert
import librosa
import numpy as np
import os
from utils import sigwin, sigrec
from scipy.io import wavfile

mixture_path = os.path.join('..', 'Datasets', 'MUSDB18')
prediction_path = os.path.join('..', 'Datasets', 'MUSDB18_predict_normed')
wiener_path = os.path.join('..', 'Datasets', 'MUSDB18_predict_normed_Wiener')

sr = 8192  # Hertz
window_length = 12  # Seconds
overlap = 75  # Percent
window_type = 'rect'

n_fft = 1024  # Frame size for spectrograms
hop_length = 768  # Hop length in samples for spectrograms

for j, folder in enumerate(sorted(os.listdir(prediction_path))):
    if not os.path.exists(os.path.join(wiener_path, folder)):
        os.mkdir(os.path.join(wiener_path, folder))
    for i, song in enumerate(sorted(os.listdir(os.path.join(prediction_path, folder)))):
        if j == 1 or (j == 0 and i >= 22):
            print(i, song)
            mixture, sr = librosa.load(os.path.join(mixture_path, folder, song, 'mixture.wav'), sr=8192, mono=True)

            vocals, _ = librosa.load(os.path.join(prediction_path, folder, song, 'vocals.wav'), mono=True, sr=8192)
            bass, _ = librosa.load(os.path.join(prediction_path, folder, song, 'bass.wav'), mono=True, sr=8192)
            drums, _ = librosa.load(os.path.join(prediction_path, folder, song, 'drums.wav'), mono=True, sr=8192)
            other, _ = librosa.load(os.path.join(prediction_path, folder, song, 'other.wav'), mono=True, sr=8192)

            print('Loading sources done!')

            mixture = mixture[:len(vocals)]
            if i == 65:
                mixture = np.pad(mixture, (0, 12 * sr - len(mixture)), 'reflect')
                vocals = np.pad(vocals, (0, 12 * sr - len(vocals)), 'reflect')
                bass = np.pad(bass, (0, 12 * sr - len(bass)), 'reflect')
                drums = np.pad(drums, (0, 12 * sr - len(drums)), 'reflect')
                other = np.pad(other, (0, 12 * sr - len(other)), 'reflect')

            mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
            bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
            drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
            vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
            other_windows = sigwin(other, window_length * sr, window_type, overlap)

            print('Windowing sources done!')

            filtered_vocals = []
            filtered_bass = []
            filtered_drums = []
            filtered_other = []

            for window_index in range(mixture_windows.shape[0]):
                mixture_window_stft = librosa.stft(mixture_windows[window_index], n_fft=n_fft, hop_length=hop_length)

                bass_window_spect = np.abs(librosa.stft(bass_windows[window_index], n_fft=n_fft,
                                                        hop_length=hop_length), dtype='float32')
                drums_window_spect = np.abs(librosa.stft(drums_windows[window_index], n_fft=n_fft,
                                                         hop_length=hop_length), dtype='float32')
                vocals_window_spect = np.abs(librosa.stft(vocals_windows[window_index], n_fft=n_fft,
                                                          hop_length=hop_length), dtype='float32')
                other_window_spect = np.abs(librosa.stft(other_windows[window_index], n_fft=n_fft,
                                                         hop_length=hop_length), dtype='float32')

                mixture_window_stft = mixture_window_stft.transpose()
                vocals_window_spect = vocals_window_spect.transpose()
                bass_window_spect = bass_window_spect.transpose()
                drums_window_spect = drums_window_spect.transpose()
                other_window_spect = other_window_spect.transpose()

                mixture_window_stft = np.expand_dims(mixture_window_stft, axis=-1)
                vocals_window_spect = np.expand_dims(vocals_window_spect, axis=(-2, -1))
                bass_window_spect = np.expand_dims(bass_window_spect, axis=(-2, -1))
                drums_window_spect = np.expand_dims(drums_window_spect, axis=(-2, -1))
                other_window_spect = np.expand_dims(other_window_spect, axis=(-2, -1))

                sources = np.concatenate((vocals_window_spect, bass_window_spect, drums_window_spect, other_window_spect), axis=-1)

                filtered_sources = norbert.wiener(sources, mixture_window_stft, iterations=1, use_softmask=False)

                filtered_vocals_spect = filtered_sources[:, :, 0, 0].transpose()
                filtered_bass_spect = filtered_sources[:, :, 0, 1].transpose()
                filtered_drums_spect = filtered_sources[:, :, 0, 2].transpose()
                filtered_other_spect = filtered_sources[:, :, 0, 3].transpose()

                filtered_vocals_window = librosa.istft(filtered_vocals_spect, hop_length=hop_length, win_length=n_fft)
                filtered_bass_window = librosa.istft(filtered_bass_spect, hop_length=hop_length, win_length=n_fft)
                filtered_drums_window = librosa.istft(filtered_drums_spect, hop_length=hop_length, win_length=n_fft)
                filtered_other_window = librosa.istft(filtered_other_spect, hop_length=hop_length, win_length=n_fft)

                filtered_vocals.append(filtered_vocals_window)
                filtered_bass.append(filtered_bass_window)
                filtered_drums.append(filtered_drums_window)
                filtered_other.append(filtered_other_window)


            print('Filtered windows done!')
            if not os.path.exists(os.path.join(wiener_path, folder, song)):
                os.mkdir(os.path.join(wiener_path, folder, song))

            new_vocals = sigrec(filtered_vocals, overlap, 'MEAN')
            new_vocals = librosa.resample(new_vocals, 8192, 44100)
            wavfile.write(os.path.join(wiener_path, folder, song, 'vocals.wav'), 44100, new_vocals)
            print('Vocals filter done!')

            new_bass = sigrec(filtered_bass, overlap, 'MEAN')
            new_bass = librosa.resample(new_bass, 8192, 44100)
            wavfile.write(os.path.join(wiener_path, folder, song, 'bass.wav'), 44100, new_bass)
            print('Bass filter done!')

            new_drums = sigrec(filtered_drums, overlap, 'MEAN')
            new_drums = librosa.resample(new_drums, 8192, 44100)
            wavfile.write(os.path.join(wiener_path, folder, song, 'drums.wav'), 44100, new_drums)
            print('Drums filter done!')

            new_other = sigrec(filtered_other, overlap, 'MEAN')
            new_other = librosa.resample(new_other, 8192, 44100)
            wavfile.write(os.path.join(wiener_path, folder, song, 'other.wav'), 44100, new_other)
            print('Other filter done!')
