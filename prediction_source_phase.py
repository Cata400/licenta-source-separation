from utils import *
import librosa
from custom_layers import ScaleInLayer, ScaleOutLayer
from custom_losses import L11_norm


def predict(test_path, model_path, multiple_sources, compute_spect, resample, sr, window_length, overlap, window_type,
            dB, n_fft, hop_length, source, normalize_from_dataset, statistics_path, normalize, normalize01, standardize,
            batch_size, save_path, network):
    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    model_name = model_path.split('\\')[-1].split('.')[0]
    song_name = test_path.split('\\')[-1].split('.')[0]

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if network.lower() == 'open_unmix':
        model = tf.keras.models.load_model(model_path, custom_objects={'ScaleInLayer': ScaleInLayer,
                                                                   'ScaleOutLayer': ScaleOutLayer})
    elif network.lower() == 'u_net':
        try:
            model = tf.keras.models.load_model(model_path)
        except:
            model = tf.keras.models.load_model(model_path, custom_objects={'L11_norm': L11_norm})

    original_sr = librosa.get_samplerate(test_path)
    if resample:
        test_song, sr = librosa.load(test_path, sr=sr, mono=True)
    else:
        test_song, sr = librosa.load(test_path, sr=original_sr, mono=True)

    if 'train' in song_name:
        orig, _ = librosa.load(os.path.join('..', 'Datasets', 'Test', 'train_1_vocals.wav'), sr=sr, mono=True)
    elif 'val' in song_name:
        orig, _ = librosa.load(os.path.join('..', 'Datasets', 'Test', 'val_10_vocals.wav'), sr=sr, mono=True)

    if not multiple_sources:
        source_windows = []
    else:
        bass_windows = []
        drums_windows = []
        vocals_windows = []
        other_windows = []

    if compute_spect:
        mixture_windows = sigwin(test_song, window_length * sr, window_type, overlap)
        orig_windows = sigwin(orig, window_length * sr, window_type, overlap)
        print('Test windows done!')

        for mixture_window, orig_window in zip(mixture_windows, orig_windows):
            mixture_spect = librosa.stft(mixture_window, n_fft=n_fft, hop_length=hop_length)
            mixture_abs = np.abs(mixture_spect)

            orig_spect = librosa.stft(orig_window, n_fft=n_fft, hop_length=hop_length)

            if network.lower() == 'u_net':
                mixture_abs = mixture_abs[:-1, :-1]

            if dB:
                reference = np.max(mixture_abs)
                mixture_abs = librosa.amplitude_to_db(mixture_abs, ref=reference)
                np.clip(mixture_abs, -80, 0, out=mixture_abs)

            maxim = np.max(mixture_abs)
            maxim_absolut = np.max(np.abs(mixture_abs))
            minim_absolut = np.min(np.abs(mixture_abs))
            minim = np.min(mixture_abs)
            mean = np.mean(mixture_abs)
            std = np.std(mixture_abs)

            if normalize_from_dataset:
                if normalize:
                    mixture_abs = mixture_abs / np.max([statistics['maxim_mixture_spect'],
                                                        np.abs(statistics['minim_mixture_spect'])])

                elif normalize01:
                    if statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect']:
                        mixture_abs = (mixture_abs - statistics['minim_mixture_spect']) / \
                                      (statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect'])
                    else:
                        mixture_abs = np.zeros(mixture_abs.shape)

                elif standardize:
                    mixture_abs = (mixture_abs - statistics['mean_mixture_spect']) / \
                                  statistics['std_mixture_spect']
            else:
                if normalize:
                    if (not dB and maxim_absolut > 10) or (dB and maxim_absolut - minim_absolut > 10):
                        mixture_abs = mixture_abs / maxim_absolut
                    else:
                        mixture_abs = np.zeros(mixture_abs.shape)

                elif normalize01:
                    if dB:
                        mixture_abs = (mixture_abs + 80) / 80
                    else:
                        if maxim - minim > 1e-2:
                            mixture_abs = (mixture_abs - minim) / (maxim - minim)
                        else:
                            mixture_abs = np.zeros(mixture_abs.shape)

                elif standardize:
                    mixture_abs = (mixture_abs - mean) / std

            if not multiple_sources:
                mixture_abs = np.expand_dims(mixture_abs, axis=0)
                source_abs = model.predict(mixture_abs, batch_size=batch_size)
                source_abs = np.squeeze(source_abs)

                if normalize_from_dataset:
                    if normalize:
                        source_abs = source_abs * np.max(statistics['maxim_' + str(source) + '_spect'],
                                                         np.abs(statistics['minim_' + str(source) + '_spect']))

                    elif normalize01:
                        source_abs = source_abs * (statistics['maxim_' + str(source) + '_spect']
                                                   - statistics['minim_' + str(source) + '_spect']) + \
                                     statistics['minim_' + str(source) + '_spect']

                    elif standardize:
                        source_abs = source_abs * statistics['std_' + str(source) + '_spect'] + \
                                     statistics['mean_' + str(source) + '_spect']
                else:
                    if normalize:
                        if (not dB and maxim_absolut > 10) or (dB and maxim_absolut - minim_absolut > 10):
                            source_abs = source_abs * maxim_absolut
                        else:
                            if dB:
                                source_abs = source_abs - maxim_absolut
                            else:
                                source_abs = source_abs + maxim_absolut

                    elif normalize01:
                        if dB:
                            source_abs = source_abs * 80 - 80
                        else:
                            if maxim - minim > 1e-2:
                                source_abs = source_abs * (maxim - minim) + minim
                            else:
                                source_abs = source_abs + maxim

                    elif standardize:
                        source_abs = source_abs * std + mean

                if dB:
                    source_abs = librosa.db_to_amplitude(source_abs, ref=reference)

                if network.lower() == 'u_net':
                    source_abs = np.r_[source_abs, np.zeros((1, source_abs.shape[1]))]

                if network.lower() == 'u_net':
                    mixture_phase = np.angle(mixture_spect[:, :-1])
                    orig_phase = np.angle(orig_spect[:, :-1])
                else:
                    mixture_phase = np.angle(mixture_spect)

                source_spect = np.multiply(source_abs, np.exp(1j * orig_phase))
                source_window = librosa.istft(source_spect, hop_length=hop_length, win_length=n_fft)
                source_windows.append(source_window)

            else:
                bass_abs, drums_abs, vocals_abs, other_abs = model.predict(mixture_abs, batch_size=batch_size)

                if normalize_from_dataset:
                    if normalize:
                        bass_abs = bass_abs * np.max(statistics['maxim_bass_spect'],
                                                     np.abs(statistics['minim_bass_spect']))
                        drums_abs = drums_abs * np.max(statistics['maxim_drums_spect'],
                                                       np.abs(statistics['minim_drums_spect']))
                        vocals_abs = vocals_abs * np.max(statistics['maxim_vocals_spect'],
                                                         np.abs(statistics['minim_vocals_spect']))
                        other_abs = other_abs * np.max(statistics['maxim_other_spect'],
                                                       np.abs(statistics['minim_other_spect']))

                    elif normalize01:
                        if statistics['maxim_bass_spect'] - statistics['minim_bass_spect']:
                            bass_abs = bass_abs * (statistics['maxim_bass_spect'] - statistics['minim_bass_spect']) + \
                                       statistics['minim_bass_spect']
                        else:
                            bass_abs = np.zeros(bass_abs.shape)
                        if statistics['maxim_drums_spect'] - statistics['minim_drums_spect']:
                            drums_abs = drums_abs * (
                                    statistics['maxim_drums_spect'] - statistics['minim_drums_spect']) + \
                                        statistics['minim_drums_spect']
                        else:
                            drums_abs = np.zeros(drums_abs.shape)
                        if statistics['maxim_vocals_spect'] - statistics['minim_vocals_spect']:
                            vocals_abs = vocals_abs * (
                                    statistics['maxim_vocals_spect'] - statistics['minim_vocals_spect']) + \
                                         statistics['minim_vocals_spect']
                        else:
                            vocals_abs = np.zeros(vocals_abs.shape)
                        if statistics['maxim_other_spect'] - statistics['minim_other_spect']:
                            other_abs = other_abs * (
                                    statistics['maxim_other_spect'] - statistics['minim_other_spect']) + \
                                        statistics['minim_other_spect']
                        else:
                            other_abs = np.zeros(other_abs.shape)

                    elif standardize:
                        bass_abs = bass_abs * statistics['std_bass_spect'] + statistics['mean_bass_spect']
                        drums_abs = drums_abs * statistics['std_drums_spect'] + statistics['mean_drum_spect']
                        vocals_abs = vocals_abs * statistics['std_vocals_spect'] + statistics['mean_vocals_spect']
                        other_abs = other_abs * statistics['std_other_spect'] + statistics['mean_other_spect']
                else:
                    if normalize:
                        if (not dB and maxim_absolut > 1e4) or (dB and maxim_absolut - minim_absolut > 10):
                            bass_abs = bass_abs * maxim_absolut
                            drums_abs = drums_abs * maxim_absolut
                            vocals_abs = vocals_abs * maxim_absolut
                            other_abs = other_abs * maxim_absolut
                        else:
                            if dB:
                                bass_abs = bass_abs - maxim_absolut
                                drums_abs = drums_abs - maxim_absolut
                                vocals_abs = vocals_abs - maxim_absolut
                                other_abs = other_abs - maxim_absolut
                            else:
                                bass_abs = bass_abs + maxim_absolut
                                drums_abs = drums_abs + maxim_absolut
                                vocals_abs = vocals_abs + maxim_absolut
                                other_abs = other_abs + maxim_absolut

                    elif normalize01:
                        if dB:
                            bass_abs = bass_abs * 80 - 80
                            drums_abs = drums_abs * 80 - 80
                            vocals_abs = vocals_abs * 80 - 80
                            other_abs = other_abs * 80 - 80
                        else:
                            if maxim - minim > 1e-4:
                                bass_abs = bass_abs * (maxim - minim) + minim
                                drums_abs = drums_abs * (maxim - minim) + minim
                                vocals_abs = vocals_abs * (maxim - minim) + minim
                                other_abs = other_abs * (maxim - minim) + minim
                            else:
                                bass_abs = bass_abs + maxim
                                drums_abs = drums_abs + maxim
                                vocals_abs = vocals_abs + maxim
                                other_abs = other_abs + maxim

                    elif standardize:
                        bass_abs = bass_abs * std + mean
                        drums_abs = drums_abs * std + mean
                        vocals_abs = vocals_abs * std + mean
                        other_abs = other_abs * std + mean

                bass_abs = librosa.db_to_amplitude(bass_abs, ref=reference)
                drums_abs = librosa.db_to_amplitude(drums_abs, ref=reference)
                vocals_abs = librosa.db_to_amplitude(vocals_abs, ref=reference)
                other_abs = librosa.db_to_amplitude(other_abs, ref=reference)

                mixture_phase = np.angle(mixture_spect)
                bass_spect = np.multiply(bass_abs, np.exp(1j * mixture_phase))
                drums_spect = np.multiply(drums_abs, np.exp(1j * mixture_phase))
                vocals_spect = np.multiply(vocals_abs, np.exp(1j * mixture_phase))
                other_spect = np.multiply(other_abs, np.exp(1j * mixture_phase))

                bass_window = librosa.istft(bass_spect, hop_length=hop_length, win_length=n_fft)
                drums_window = librosa.istft(drums_spect, hop_length=hop_length, win_length=n_fft)
                vocals_window = librosa.istft(vocals_spect, hop_length=hop_length, win_length=n_fft)
                other_window = librosa.istft(other_spect, hop_length=hop_length, win_length=n_fft)

                bass_windows.append(bass_window)
                drums_windows.append(drums_window)
                vocals_windows.append(vocals_window)
                other_windows.append(other_window)

        if not multiple_sources:
            print('Prediction windows done!')
            if window_type == 'rect':
                test_source = sigrec(source_windows, overlap, 'MEAN')
            else:
                test_source = sigrec(source_windows, overlap, 'OLA')
            wavfile.write(save_path.split('.wav')[0] + '_' + model_name + '_' + str(source) + '.wav', sr, test_source)
            print(str(source).capitalize() + ' prediction done!')
        else:
            print('Prediction windows done!')

            test_bass = sigrec(bass_windows, overlap, 'OLA')
            wavfile.write(save_path + model_name + '_bass.wav', sr, test_bass)
            print('Bass prediction done!')

            test_drums = sigrec(drums_windows, overlap, 'OLA')
            wavfile.write(save_path + model_name + '_drums.wav', sr, test_drums)
            print('Drums prediction done!')

            test_vocals = sigrec(vocals_windows, overlap, 'OLA')
            wavfile.write(save_path + model_name + '_vocals.wav', sr, test_vocals)
            print('Vocals prediction done!')

            test_other = sigrec(other_windows, overlap, 'OLA')
            wavfile.write(save_path + model_name + '_other.wav', sr, test_other)
            print('Other prediction done!')
    else:
        maxim = np.max(test_song)
        maxim_absolut = np.max(np.abs(test_song))
        minim = np.min(test_song)
        mean = np.mean(test_song)
        std = np.std(test_song)

        if normalize_from_dataset:
            if normalize:
                test_song = test_song / np.max([statistics['maxim_mixture'], np.abs(statistics['minim_mixture'])])

            elif normalize01:
                test_song = (test_song - statistics['minim_mixture']) / (statistics['maxim_mixture'] -
                                                                         statistics['minim_mixture'])

            elif standardize:
                test_song = (test_song - statistics['mean_mixture']) / statistics['std_mixture']
        else:
            if normalize:
                test_song = test_song / maxim_absolut

            elif normalize01:
                test_song = (test_song - minim) / (maxim - minim)

            elif standardize:
                test_song = (test_song - mean) / std

        mixture_windows = sigwin(test_song, window_length * sr, window_type, overlap)
        print('Test windows done!')

        for mixture_window in mixture_windows:
            if not multiple_sources:
                source_window = model.predict(mixture_window, batch_size=batch_size)
                source_windows.append(source_window)
            else:
                bass_window, drums_window, vocals_window, other_window = model.predict(mixture_window,
                                                                                       batch_size=batch_size)
                bass_windows.append(bass_window)
                drums_windows.append(drums_window)
                vocals_windows.append(vocals_window)
                other_windows.append(other_window)

        if not multiple_sources:
            print('Prediction windows done!')
            test_source = sigrec(source_windows, overlap, 'OLA')

            if normalize_from_dataset:
                if normalize:
                    test_source = test_source * np.max([statistics['maxim_' + str(source)],
                                                        np.abs(statistics['minim_' + str(source)])])

                elif normalize01:
                    test_source = test_source * (statistics['maxim_' + str(source)] -
                                                 statistics['minim_' + str(source)]) + \
                                  statistics['minim_' + str(source)]

                elif standardize:
                    test_source = test_source * statistics['std_' + str(source)] + \
                                  statistics['mean_' + str(source)]
            else:
                if normalize:
                    test_source = test_source * maxim_absolut

                elif normalize01:
                    test_source = test_source * (maxim - minim) + minim

                elif standardize:
                    test_source = test_source * std + mean

            wavfile.write(save_path + '_' + str(source) + '.wav', sr, test_source)
            print(str(source).capitalize() + ' prediction done!')

        else:
            print('Prediction windows done!')

            test_bass = sigrec(bass_windows, overlap, 'OLA')
            test_drums = sigrec(drums_windows, overlap, 'OLA')
            test_vocals = sigrec(vocals_windows, overlap, 'OLA')
            test_other = sigrec(other_windows, overlap, 'OLA')

            if normalize_from_dataset:
                if normalize:
                    test_bass = test_bass * np.max([statistics['maxim_bass'], np.abs(statistics['minim_bass'])])
                    test_drums = test_drums * np.max([statistics['maxim_drums'], np.abs(statistics['minim_drums'])])
                    test_vocals = test_vocals * np.max([statistics['maxim_vocals'], np.abs(statistics['minim_vocals'])])
                    test_other = test_other * np.max([statistics['maxim_other'], np.abs(statistics['minim_other'])])

                elif normalize01:
                    test_bass = test_bass * (statistics['maxim_bass'] - statistics['minim_bass']) + \
                                statistics['minim_bass']
                    test_drums = test_drums * (statistics['maxim_drums'] - statistics['minim_drums']) + \
                                 statistics['minim_drums']
                    test_vocals = test_vocals * (statistics['maxim_vocals'] - statistics['minim_vocals']) + \
                                  statistics['minim_vocals']
                    test_other = test_other * (statistics['maxim_other'] - statistics['minim_other']) + \
                                 statistics['minim_other']

                elif standardize:
                    test_bass = test_bass * statistics['std_bass'] + statistics['mean_bass']
                    test_drums = test_drums * statistics['std_drums'] + statistics['mean_drums']
                    test_vocals = test_vocals * statistics['std_vocals'] + statistics['mean_vocals']
                    test_other = test_other * statistics['std_other'] + statistics['mean_other']
            else:
                if normalize:
                    test_bass = test_bass * maxim_absolut
                    test_drums = test_drums * maxim_absolut
                    test_vocals = test_vocals * maxim_absolut
                    test_other = test_other * maxim_absolut

                elif normalize01:
                    test_bass = test_bass * (maxim - minim) + minim
                    test_drums = test_drums * (maxim - minim) + minim
                    test_vocals = test_vocals * (maxim - minim) + minim
                    test_other = test_other * (maxim - minim) + minim

                elif standardize:
                    test_bass = test_bass * std + mean
                    test_drums = test_drums * std + mean
                    test_vocals = test_vocals * std + mean
                    test_other = test_other * std + mean

            wavfile.write(save_path + '_bass.wav', sr, test_bass)
            wavfile.write(save_path + '_drums.wav', sr, test_drums)
            wavfile.write(save_path + '_vocals.wav', sr, test_vocals)
            wavfile.write(save_path + '_other.wav', sr, test_other)
            print('All predictions done!')
