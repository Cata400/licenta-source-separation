from utils import *


def preprocess_1_source(dataset, resample, sr, window_length, overlap, window_type, compute_spect, dB, n_fft,
                        hop_length,
                        writer_train, writer_val, source, extra_song_shuffle, intra_song_shuffle,
                        normalize_from_dataset,
                        statistics_path, normalize, normalize01, standardize, card_txt, network):
    train_card, val_card = 0, 0
    np.random.seed(42)

    txt = ''
    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if dataset.lower() == 'musdb18':
        for subfolder in sorted(os.listdir(os.path.join('..', 'Datasets', 'MUSDB18'))):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            if extra_song_shuffle:
                np.random.shuffle(songs)
            else:
                songs = sorted(songs)

            for i, song in enumerate(songs):
                # if i == 2:
                #     break
                print(i, song)
                if resample:
                    x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)
                else:
                    sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
                    x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)

                if compute_spect:
                    x_windows = sigwin(x, window_length * sr, window_type, overlap)
                    y_windows = sigwin(y, window_length * sr, window_type, overlap)

                    if intra_song_shuffle:
                        np.random.shuffle(x_windows)
                        np.random.shuffle(y_windows)

                    for window_index in range(x_windows.shape[0]):
                        x_window_spect = np.abs(librosa.stft(x_windows[window_index], n_fft=n_fft,
                                                             hop_length=hop_length), dtype='float32')
                        y_window_spect = np.abs(librosa.stft(y_windows[window_index], n_fft=n_fft,
                                                             hop_length=hop_length), dtype='float32')

                        if network.lower() == 'u_net':
                            x_window_spect = x_window_spect[:-1, :-1]
                            y_window_spect = y_window_spect[:-1, :-1]

                        if dB:
                            y_window_spect = librosa.amplitude_to_db(y_window_spect, ref=np.max(x_window_spect))
                            x_window_spect = librosa.amplitude_to_db(x_window_spect, ref=np.max)

                            np.clip(x_window_spect, -80, 0, x_window_spect)
                            np.clip(y_window_spect, -80, 0, y_window_spect)

                        if normalize_from_dataset:
                            if normalize:
                                x_window_spect = x_window_spect / np.max([statistics['maxim_mixture_spect'],
                                                                          np.abs(statistics['minim_mixture_spect'])])
                                y_window_spect = y_window_spect / np.max([statistics['maxim_' + str(source) +
                                                                                     '_spect'], np.abs(
                                    statistics['minim_' + str(source) + '_spect'])])

                            elif normalize01:
                                if statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect']:
                                    x_window_spect = (x_window_spect - statistics['minim_mixture_spect']) / \
                                                     (statistics['maxim_mixture_spect'] - statistics[
                                                         'minim_mixture_spect'])
                                else:
                                    x_window_spect = np.zeros(x_window_spect.shape)
                                if statistics['maxim_' + str(source) + '_spect'] - \
                                        statistics['minim_' + str(source) + '_spect']:
                                    y_window_spect = (y_window_spect - statistics['minim_' + str(source) + '_spect']) / \
                                                     (statistics['maxim_' + str(source) + '_spect'] -
                                                      statistics['minim_' + str(source) + '_spect'])
                                else:
                                    y_window_spect = np.zeros(y_window_spect)

                            elif standardize:
                                x_window_spect = (x_window_spect - statistics['mean_mixture_spect']) / \
                                                 statistics['std_mixture_spect']
                                y_window_spect = (y_window_spect - statistics['mean_' + str(source) + '_spect']) / \
                                                 statistics['std_' + str(source) + '_spect']
                        else:
                            if normalize:
                                if (not dB and np.max(np.abs(x_window_spect)) > 10) or (
                                        dB and np.max(np.abs(x_window_spect)) - np.min(np.abs(x_window_spect)) > 10):
                                    x_window_spect = x_window_spect / np.max(np.abs(x_window_spect))
                                else:
                                    x_window_spect = np.zeros(x_window_spect.shape)
                                if (not dB and np.max(np.abs(y_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(y_window_spect)) - np.min(np.abs(y_window_spect)) > 10):
                                    y_window_spect = y_window_spect / np.max(np.abs(y_window_spect))
                                else:
                                    y_window_spect = np.zeros(y_window_spect.shape)

                            elif normalize01:
                                if dB:
                                    x_window_spect = (x_window_spect + 80) / 80
                                    y_window_spect = (y_window_spect + 80) / 80
                                else:
                                    if np.max(x_window_spect) - np.min(x_window_spect) > 1e-2:
                                        x_window_spect = (x_window_spect - np.min(x_window_spect)) / \
                                                         (np.max(x_window_spect) - np.min(x_window_spect))
                                    else:
                                        x_window_spect = np.zeros(x_window_spect.shape)
                                    if np.max(y_window_spect) - np.min(y_window_spect) > 1e-2:
                                        y_window_spect = (y_window_spect - np.min(y_window_spect)) / \
                                                         (np.max(y_window_spect) - np.min(y_window_spect))
                                    else:
                                        y_window_spect = np.zeros(y_window_spect.shape)

                            elif standardize:
                                x_window_spect = (x_window_spect - np.mean(x_window_spect)) / np.std(x_window_spect)
                                y_window_spect = (y_window_spect - np.mean(y_window_spect)) / np.std(y_window_spect)

                        #     if window_index == x_windows.shape[0] // 2:
                        #         print(x_window_spect.shape, y_window_spect.shape)
                        #         print('extreme x:', np.max(x_window_spect), np.min(x_window_spect))
                        #         print('extreme y:', np.max(y_window_spect), np.min(y_window_spect))
                        #         print('stats x:', np.mean(x_window_spect), np.std(x_window_spect))
                        #         print('stats y:', np.mean(y_window_spect), np.std(y_window_spect))
                        #           break

                        if subfolder == 'train':
                            train_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                                    y_window_spect.astype(np.float32))
                            writer_train.write(train_serialized_spectrograms)
                            train_card += 1
                        elif subfolder == 'val':
                            val_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                                  y_window_spect.astype(np.float32))
                            writer_val.write(val_serialized_spectrograms)
                            val_card += 1
                        txt += song + ' ' + str(x_window_spect.dtype) + ' ' + str(y_window_spect.dtype) + '\n'


                else:
                    if normalize_from_dataset:
                        if normalize:
                            x = x / np.max([statistics['maxim_mixture'], np.abs(statistics['minim_mixture'])])
                            y = y / np.max([statistics['maxim_' + str(source)], np.abs(statistics['minim_'
                                                                                                  + str(source)])])

                        elif normalize01:
                            x = (x - statistics['minim_mixture']) / (statistics['maxim_mixture']
                                                                     - statistics['minim_mixture'])
                            y = (y - statistics['minim_' + str(source)]) / (statistics['maxim_' + str(source)] -
                                                                            statistics['minim_' + str(source)])

                        elif standardize:
                            x = (x - statistics['mean_mixture']) / statistics['std_mixture']
                            y = (y - statistics['mean_' + str(source)]) / statistics['std_' + str(source)]
                    else:
                        if normalize:
                            x = x / np.max(np.abs(x))
                            y = y / np.max(np.abs(y))

                        elif normalize01:
                            x = (x - np.min(x)) / (np.max(x) - np.min(x))
                            y = (y - np.min(y)) / (np.max(y) - np.min(y))

                        elif standardize:
                            x = (x - np.mean(x)) / np.std(x)
                            y = (y - np.mean(y)) / np.std(y)

                    x_windows = sigwin(x, window_length * sr, window_type, overlap)
                    y_windows = sigwin(y, window_length * sr, window_type, overlap)

                    if intra_song_shuffle:
                        np.random.shuffle(x_windows)
                        np.random.shuffle(y_windows)

                    # print(x_windows.shape, y_windows.shape)
                    # print('extreme x:', np.max(x_windows), np.min(x_windows))
                    # print('extreme y:', np.max(y_windows), np.min(y_windows))
                    # print('stats x:', np.mean(x_windows), np.std(x_windows))
                    # print('stats y:', np.mean(y_windows), np.std(y_windows))
                    # break

                    for window_index in range(x_windows.shape[0]):
                        if subfolder == 'train':
                            train_serialized_waveforms = serialize_data_1_source(x_windows[window_index],
                                                                                 y_windows[window_index])
                            writer_train.write(train_serialized_waveforms)
                            train_card += 1
                        elif subfolder == 'val':
                            val_serialized_waveforms = serialize_data_1_source(x_windows[window_index],
                                                                               y_windows[window_index])
                            writer_val.write(val_serialized_waveforms)
                            val_card += 1

    else:
        raise Exception('Dataset is not correct')

    with open('out.txt', 'w') as f:
        print(txt, file=f)

    write_cardinality(os.path.join('..', 'Cardinality', card_txt), train_card, val_card)


def preprocess_all_sources(dataset, resample, sr, window_length, overlap, window_type, compute_spect, dB, n_fft,
                           hop_length, writer_train, writer_val, extra_song_shuffle, intra_song_shuffle,
                           normalize_from_dataset, statistics_path, normalize, normalize01, standardize, card_txt):
    train_card, val_card = 0, 0
    np.random.seed(42)

    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if dataset.lower() == 'musdb18':
        for subfolder in sorted(os.listdir(os.path.join('..', 'Datasets', 'MUSDB18'))):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            if extra_song_shuffle:
                np.random.shuffle(songs)
            else:
                songs = sorted(songs)

            for i, song in enumerate(songs):
                print(i, song)
                if resample:
                    mixture, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)
                else:
                    sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
                    mixture, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)

                if compute_spect:
                    mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
                    bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
                    drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
                    vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
                    other_windows = sigwin(other, window_length * sr, window_type, overlap)

                    if intra_song_shuffle:
                        np.random.shuffle(mixture_windows)
                        np.random.shuffle(bass_windows)
                        np.random.shuffle(drums_windows)
                        np.random.shuffle(vocals_windows)
                        np.random.shuffle(other_windows)

                    for window_index in range(mixture_windows.shape[0]):
                        mixture_window_spect = np.abs(librosa.stft(mixture_windows[window_index], n_fft=n_fft,
                                                                   hop_length=hop_length), dtype='float32')
                        bass_window_spect = np.abs(librosa.stft(bass_windows[window_index], n_fft=n_fft,
                                                                hop_length=hop_length), dtype='float32')
                        drums_window_spect = np.abs(librosa.stft(drums_windows[window_index], n_fft=n_fft,
                                                                 hop_length=hop_length), dtype='float32')
                        vocals_window_spect = np.abs(librosa.stft(vocals_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')
                        other_window_spect = np.abs(librosa.stft(other_windows[window_index], n_fft=n_fft,
                                                                 hop_length=hop_length), dtype='float32')

                        if dB:
                            bass_window_spect = librosa.amplitude_to_db(bass_window_spect,
                                                                        ref=np.max(mixture_window_spect))
                            drums_window_spect = librosa.amplitude_to_db(drums_window_spect,
                                                                         ref=np.max(mixture_window_spect))
                            vocals_window_spect = librosa.amplitude_to_db(vocals_window_spect,
                                                                          ref=np.max(mixture_window_spect))
                            other_window_spect = librosa.amplitude_to_db(other_window_spect,
                                                                         ref=np.max(mixture_window_spect))
                            mixture_window_spect = librosa.amplitude_to_db(mixture_window_spect, ref=np.max)

                            np.clip(mixture_window_spect, -80, 0, mixture_window_spect)
                            np.clip(bass_window_spect, -80, 0, bass_window_spect)
                            np.clip(drums_window_spect, -80, 0, drums_window_spect)
                            np.clip(vocals_window_spect, -80, 0, vocals_window_spect)
                            np.clip(other_window_spect, -80, 0, other_window_spect)

                        if normalize_from_dataset:
                            if normalize:
                                mixture_window_spect = mixture_window_spect / np.max([statistics['maxim_mixture_spect'],
                                                                                      np.abs(statistics[
                                                                                                 'minim_mixture_spect'])])
                                bass_window_spect = bass_window_spect / np.max([statistics['maxim_bass_spect'],
                                                                                np.abs(statistics['minim_bass_spect'])])
                                drums_window_spect = drums_window_spect / np.max([statistics['maxim_drums_spect'],
                                                                                  np.abs(
                                                                                      statistics['minim_drums_spect'])])
                                vocals_window_spect = vocals_window_spect / np.max([statistics['maxim_vocals_spect'],
                                                                                    np.abs(statistics[
                                                                                               'minim_vocals_spect'])])
                                other_window_spect = other_window_spect / np.max([statistics['maxim_other_spect'],
                                                                                  np.abs(
                                                                                      statistics['minim_other_spect'])])

                            elif normalize01:
                                if statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect']:
                                    mixture_window_spect = (mixture_window_spect - statistics['minim_mixture_spect']) / \
                                                           (statistics['maxim_mixture_spect'] - statistics[
                                                               'minim_mixture_spect'])
                                else:
                                    mixture_window_spect = np.zeros(mixture_window_spect)
                                if statistics['maxim_bass_spect'] - statistics['minim_bass_spect']:
                                    bass_window_spect = (bass_window_spect - statistics['minim_bass_spect']) / \
                                                        (statistics['maxim_bass_spect'] - statistics[
                                                            'minim_bass_spect'])
                                else:
                                    bass_window_spect = np.zeros(bass_window_spect)
                                if statistics['maxim_drums_spect'] - statistics['minim_drums_spect']:
                                    drums_window_spect = (drums_window_spect - statistics['minim_drums_spect']) / \
                                                         (statistics['maxim_drums_spect'] - statistics[
                                                             'minim_drums_spect'])
                                else:
                                    drums_window_spect = np.zeros(drums_window_spect)
                                if statistics['maxim_vocals_spect'] - statistics['minim_vocals_spect']:
                                    vocals_window_spect = (vocals_window_spect - statistics['minim_vocals_spect']) / \
                                                          (statistics['maxim_vocals_spect'] - statistics[
                                                              'minim_vocals_spect'])
                                else:
                                    vocals_window_spect = np.zeros(vocals_window_spect)
                                if statistics['maxim_other_spect'] - statistics['minim_other_spect']:
                                    other_window_spect = (other_window_spect - statistics['minim_other_spect']) / \
                                                         (statistics['maxim_other_spect'] - statistics[
                                                             'minim_other_spect'])
                                else:
                                    other_window_spect = np.zeros(other_window_spect)

                            elif standardize:
                                mixture_window_spect = (mixture_window_spect - statistics['mean_mixture_spect']) / \
                                                       statistics['std_mixture_spect']
                                bass_window_spect = (bass_window_spect - statistics['mean_bass_spect']) / \
                                                    statistics['std_bass_spect']
                                drums_window_spect = (drums_window_spect - statistics['mean_drums_spect']) / \
                                                     statistics['std_drums_spect']
                                vocals_window_spect = (vocals_window_spect - statistics['mean_vocals_spect']) / \
                                                      statistics['std_vocals_spect']
                                other_window_spect = (other_window_spect - statistics['mean_other_spect']) / \
                                                     statistics['std_other_spect']
                        else:
                            if normalize:
                                if (not dB and np.max(np.abs(mixture_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(mixture_window_spect)) -
                                        np.min(np.abs(mixture_window_spect)) > 10):
                                    mixture_window_spect = mixture_window_spect / np.max(np.abs(mixture_window_spect))
                                else:
                                    mixture_window_spect = np.zeros(mixture_window_spect.shape)
                                if (not dB and np.max(np.abs(bass_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(bass_window_spect)) -
                                        np.min(np.abs(bass_window_spect)) > 10):
                                    bass_window_spect = bass_window_spect / np.max(np.abs(bass_window_spect))
                                else:
                                    bass_window_spect = np.zeros(bass_window_spect.shape)
                                if (not dB and np.max(np.abs(drums_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(drums_window_spect)) -
                                        np.min(np.abs(drums_window_spect)) > 10):
                                    drums_window_spect = drums_window_spect / np.max(np.abs(drums_window_spect))
                                else:
                                    drums_window_spect = np.zeros(drums_window_spect.shape)
                                if (not dB and np.max(np.abs(vocals_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(vocals_window_spect)) -
                                        np.min(np.abs(vocals_window_spect)) > 10):
                                    vocals_window_spect = vocals_window_spect / np.max(np.abs(vocals_window_spect))
                                else:
                                    vocals_window_spect = np.zeros(vocals_window_spect.shape)
                                if (not dB and np.max(np.abs(other_window_spect)) > 1e4) or (
                                        dB and np.max(np.abs(other_window_spect)) -
                                        np.min(np.abs(other_window_spect)) > 10):
                                    other_window_spect = other_window_spect / np.max(np.abs(other_window_spect))
                                else:
                                    other_window_spect = np.zeros(other_window_spect.shape)

                            elif normalize01:
                                if dB:
                                    mixture_window_spect = (mixture_window_spect + 80) / 80
                                    bass_window_spect = (bass_window_spect + 80) / 80
                                    drums_window_spect = (drums_window_spect + 80) / 80
                                    vocals_window_spect = (vocals_window_spect + 80) / 80
                                    other_window_spect = (other_window_spect + 80) / 80
                                else:
                                    if np.max(mixture_window_spect) - np.min(mixture_window_spect) > 1e4:
                                        mixture_window_spect = (mixture_window_spect - np.min(mixture_window_spect)) / \
                                                               (np.max(mixture_window_spect) - np.min(
                                                                   mixture_window_spect))
                                    else:
                                        mixture_window_spect = np.zeros(mixture_window_spect.shape)
                                    if np.max(bass_window_spect) - np.min(bass_window_spect) > 1e4:
                                        bass_window_spect = (bass_window_spect - np.min(bass_window_spect)) / \
                                                            (np.max(bass_window_spect) - np.min(bass_window_spect))
                                    else:
                                        bass_window_spect = np.zeros(bass_window_spect.shape)
                                    if np.max(drums_window_spect) - np.min(bass_window_spect) > 1e4:
                                        drums_window_spect = (drums_window_spect - np.min(drums_window_spect)) / \
                                                             (np.max(drums_window_spect) - np.min(drums_window_spect))
                                    else:
                                        drums_window_spect = np.zeros(drums_window_spect.shape)
                                    if np.max(vocals_window_spect) - np.min(vocals_window_spect) > 1e4:
                                        vocals_window_spect = (vocals_window_spect - np.min(vocals_window_spect)) / \
                                                              (np.max(vocals_window_spect) - np.min(
                                                                  vocals_window_spect))
                                    else:
                                        vocals_window_spect = np.zeros(vocals_window_spect.shape)
                                    if np.max(other_window_spect) - np.min(other_window_spect) > 1e4:
                                        other_window_spect = (other_window_spect - np.min(other_window_spect)) / \
                                                             (np.max(other_window_spect) - np.min(other_window_spect))
                                    else:
                                        other_window_spect = np.zeros(other_window_spect.shape)

                            elif standardize:
                                mixture_window_spect = (mixture_window_spect - np.mean(mixture_window_spect)) / \
                                                       np.std(mixture_window_spect)
                                bass_window_spect = (bass_window_spect - np.mean(bass_window_spect)) / \
                                                    np.std(bass_window_spect)
                                drums_window_spect = (drums_window_spect - np.mean(drums_window_spect)) / \
                                                     np.std(drums_window_spect)
                                vocals_window_spect = (vocals_window_spect - np.mean(vocals_window_spect)) / \
                                                      np.std(vocals_window_spect)
                                other_window_spect = (other_window_spect - np.mean(other_window_spect)) / \
                                                     np.std(other_window_spect)

                        #     if window_index == x_windows.shape[0] // 2:
                        #         print(x_window_spect.shape, y_window_spect.shape)
                        #         print('extreme x:', np.max(x_window_spect), np.min(x_window_spect))
                        #         print('extreme y:', np.max(y_window_spect), np.min(y_window_spect))
                        #         print('stats x:', np.mean(x_window_spect), np.std(x_window_spect))
                        #         print('stats y:', np.mean(y_window_spect), np.std(y_window_spect))
                        #           break

                        if subfolder == 'train':
                            train_serialized_spectrograms = serialize_data_all_sources(mixture_window_spect,
                                                                                       bass_window_spect,
                                                                                       drums_window_spect,
                                                                                       vocals_window_spect,
                                                                                       other_window_spect)
                            writer_train.write(train_serialized_spectrograms)
                            train_card += 1
                        elif subfolder == 'val':
                            val_serialized_spectrograms = serialize_data_all_sources(mixture_window_spect,
                                                                                     bass_window_spect,
                                                                                     drums_window_spect,
                                                                                     vocals_window_spect,
                                                                                     other_window_spect)
                            writer_val.write(val_serialized_spectrograms)
                            val_card += 1

                else:
                    if normalize_from_dataset:
                        if normalize:
                            mixture = mixture / np.max(
                                [statistics['maxim_mixture'], np.abs(statistics['minim_mixture'])])
                            bass = bass / np.max([statistics['maxim_bass'], np.abs(statistics['minim_bass'])])
                            drums = drums / np.max([statistics['maxim_drums'], np.abs(statistics['minim_drums'])])
                            vocals = vocals / np.max([statistics['maxim_vocals'], np.abs(statistics['minim_vocals'])])
                            other = other / np.max([statistics['maxim_other'], np.abs(statistics['minim_other'])])

                        elif normalize01:
                            mixture = (mixture - statistics['minim_mixture']) / (statistics['maxim_mixture'] -
                                                                                 statistics['minim_mixture'])
                            bass = (bass - statistics['minim_bass']) / (statistics['maxim_bass'] -
                                                                        statistics['minim_bass'])
                            drums = (drums - statistics['minim_drums']) / (statistics['maxim_drums'] -
                                                                           statistics['minim_drums'])
                            vocals = (vocals - statistics['minim_vocals']) / (statistics['maxim_vocals'] -
                                                                              statistics['minim_vocals'])
                            other = (other - statistics['minim_other']) / (statistics['maxim_other'] -
                                                                           statistics['minim_other'])

                        elif standardize:
                            mixture = (mixture - statistics['mean_mixture']) / statistics['std_mixture']
                            bass = (bass - statistics['mean_bass']) / statistics['std_bass']
                            drums = (drums - statistics['mean_drums']) / statistics['std_drums']
                            vocals = (vocals - statistics['mean_vocals']) / statistics['std_vocals']
                            other = (other - statistics['mean_other']) / statistics['std_other']
                    else:
                        if normalize:
                            mixture = mixture / np.max(np.abs(mixture))
                            bass = bass / np.max(np.abs(bass))
                            drums = drums / np.max(np.abs(drums))
                            vocals = vocals / np.max(np.abs(vocals))
                            other = other / np.max(np.abs(other))

                        elif normalize01:
                            mixture = (mixture - np.min(mixture)) / (np.max(mixture) - np.min(mixture))
                            bass = (bass - np.min(bass)) / (np.max(bass) - np.min(bass))
                            drums = (drums - np.min(drums)) / (np.max(drums) - np.min(drums))
                            vocals = (vocals - np.min(vocals)) / (np.max(vocals) - np.min(vocals))
                            other = (other - np.min(other)) / (np.max(other) - np.min(other))

                        elif standardize:
                            mixture = (mixture - np.mean(mixture)) / np.std(mixture)
                            bass = (bass - np.mean(bass)) / np.std(bass)
                            drums = (drums - np.mean(drums)) / np.std(drums)
                            vocals = (vocals - np.mean(vocals)) / np.std(vocals)
                            other = (other - np.mean(other)) / np.std(other)

                    mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
                    bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
                    drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
                    vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
                    other_windows = sigwin(other, window_length * sr, window_type, overlap)

                    if intra_song_shuffle:
                        np.random.shuffle(mixture_windows)
                        np.random.shuffle(bass_windows)
                        np.random.shuffle(drums_windows)
                        np.random.shuffle(vocals_windows)
                        np.random.shuffle(other_windows)

                    # print(x_windows.shape, y_windows.shape)
                    # print('extreme x:', np.max(x_windows), np.min(x_windows))
                    # print('extreme y:', np.max(y_windows), np.min(y_windows))
                    # print('stats x:', np.mean(x_windows), np.std(x_windows))
                    # print('stats y:', np.mean(y_windows), np.std(y_windows))
                    # break

                    for window_index in range(mixture_windows.shape[0]):
                        if subfolder == 'train':
                            train_serialized_waveforms = serialize_data_all_sources(mixture_windows[window_index],
                                                                                    bass_windows[window_index],
                                                                                    drums_windows[window_index],
                                                                                    vocals_windows[window_index],
                                                                                    other_windows[window_index])
                            writer_train.write(train_serialized_waveforms)
                            train_card += 1
                        elif subfolder == 'val':
                            val_serialized_waveforms = serialize_data_all_sources(mixture_windows[window_index],
                                                                                  bass_windows[window_index],
                                                                                  drums_windows[window_index],
                                                                                  vocals_windows[window_index],
                                                                                  other_windows[window_index])
                            writer_val.write(val_serialized_waveforms)
                            val_card += 1

    else:
        raise Exception('Dataset is not correct!')

    write_cardinality(os.path.join('..', 'Cardinality', card_txt), train_card, val_card)


def preprocess_inverse(dataset, resample, sr, window_length, overlap, window_type, compute_spect, dB, n_fft,
                       hop_length, writer_train, writer_val, extra_song_shuffle, intra_song_shuffle,
                       normalize_from_dataset, statistics_path, normalize, normalize01, standardize, card_txt):
    train_card, val_card = 0, 0
    np.random.seed(42)

    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if dataset.lower() == 'musdb18':
        for subfolder in sorted(os.listdir(os.path.join('..', 'Datasets', 'MUSDB18'))):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            if extra_song_shuffle:
                np.random.shuffle(songs)
            else:
                songs = sorted(songs)

            for i, song in enumerate(songs):
                print(i, song)
                if resample:
                    mixture, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)
                else:
                    sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
                    mixture, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)

                if compute_spect:
                    mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
                    bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
                    drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
                    vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
                    other_windows = sigwin(other, window_length * sr, window_type, overlap)

                    if intra_song_shuffle:
                        np.random.shuffle(mixture_windows)
                        np.random.shuffle(bass_windows)
                        np.random.shuffle(drums_windows)
                        np.random.shuffle(vocals_windows)
                        np.random.shuffle(other_windows)

                    for window_index in range(mixture_windows.shape[0]):
                        mixture_window_spect = np.abs(librosa.stft(mixture_windows[window_index], n_fft=n_fft,
                                                                   hop_length=hop_length), dtype='float32')
                        bass_window_spect = np.abs(librosa.stft(bass_windows[window_index], n_fft=n_fft,
                                                                hop_length=hop_length), dtype='float32')
                        drums_window_spect = np.abs(librosa.stft(drums_windows[window_index], n_fft=n_fft,
                                                                 hop_length=hop_length), dtype='float32')
                        vocals_window_spect = np.abs(librosa.stft(vocals_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')
                        other_window_spect = np.abs(librosa.stft(other_windows[window_index], n_fft=n_fft,
                                                                 hop_length=hop_length), dtype='float32')

                        x_window_spect = mixture_window_spect
                        y_window_spect = bass_window_spect + drums_window_spect + other_window_spect

                        if dB:
                            y_window_spect = librosa.amplitude_to_db(y_window_spect, ref=np.max(x_window_spect))
                            x_window_spect = librosa.amplitude_to_db(x_window_spect, ref=np.max)

                            np.clip(x_window_spect, -80, 0, x_window_spect)
                            np.clip(y_window_spect, -80, 0, y_window_spect)

                        if normalize:
                            if (not dB and np.max(np.abs(x_window_spect)) > 10) or (
                                    dB and np.max(np.abs(x_window_spect)) - np.min(np.abs(x_window_spect)) > 10):
                                x_window_spect = x_window_spect / np.max(np.abs(x_window_spect))
                            else:
                                x_window_spect = np.zeros(x_window_spect.shape)
                            if (not dB and np.max(np.abs(y_window_spect)) > 1e4) or (
                                    dB and np.max(np.abs(y_window_spect)) - np.min(np.abs(y_window_spect)) > 10):
                                y_window_spect = y_window_spect / np.max(np.abs(y_window_spect))
                            else:
                                y_window_spect = np.zeros(y_window_spect.shape)

                        elif normalize01:
                            if dB:
                                x_window_spect = (x_window_spect + 80) / 80
                                y_window_spect = (y_window_spect + 80) / 80
                            else:
                                if np.max(x_window_spect) - np.min(x_window_spect) > 1e-2:
                                    x_window_spect = (x_window_spect - np.min(x_window_spect)) / \
                                                     (np.max(x_window_spect) - np.min(x_window_spect))
                                else:
                                    x_window_spect = np.zeros(x_window_spect.shape)
                                if np.max(y_window_spect) - np.min(y_window_spect) > 1e-2:
                                    y_window_spect = (y_window_spect - np.min(y_window_spect)) / \
                                                     (np.max(y_window_spect) - np.min(y_window_spect))
                                else:
                                    y_window_spect = np.zeros(y_window_spect.shape)

                        elif standardize:
                            x_window_spect = (x_window_spect - np.mean(x_window_spect)) / np.std(x_window_spect)
                            y_window_spect = (y_window_spect - np.mean(y_window_spect)) / np.std(y_window_spect)

                        #     if window_index == x_windows.shape[0] // 2:
                        #         print(x_window_spect.shape, y_window_spect.shape)
                        #         print('extreme x:', np.max(x_window_spect), np.min(x_window_spect))
                        #         print('extreme y:', np.max(y_window_spect), np.min(y_window_spect))
                        #         print('stats x:', np.mean(x_window_spect), np.std(x_window_spect))
                        #         print('stats y:', np.mean(y_window_spect), np.std(y_window_spect))
                        #           break
                        if subfolder == 'train':
                            train_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                                    y_window_spect.astype(np.float32))
                            writer_train.write(train_serialized_spectrograms)
                            train_card += 1
                        elif subfolder == 'val':
                            val_serialized_spectrograms = serialize_data_1_source(x_window_spect.astype(np.float32),
                                                                                  y_window_spect.astype(np.float32))
                            writer_val.write(val_serialized_spectrograms)
                            val_card += 1


def preprocess_1_source_aug(dataset, resample, sr, window_length, overlap, window_type, compute_spect, dB, n_fft,
                            hop_length, writer_train, writer_val, source, extra_song_shuffle, intra_song_shuffle,
                            normalize_from_dataset, statistics_path, normalize, normalize01, standardize, card_txt,
                            network, augments):
    train_card, val_card = 0, 0
    np.random.seed(42)

    txt = ''
    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if dataset.lower() == 'musdb18':
        for subfolder in sorted(os.listdir(os.path.join('..', 'Datasets', 'MUSDB18'))):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            if extra_song_shuffle:
                np.random.shuffle(songs)
            else:
                songs = sorted(songs)

            for i, song in enumerate(songs):
                # if i == 1:
                #     break
                print(i, song)
                if resample:
                    x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)
                else:
                    sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
                    x, _ = librosa.load(os.path.join(path, song, 'mixture.wav'), sr=sr, mono=True)
                    y, _ = librosa.load(os.path.join(path, song, source + '.wav'), sr=sr, mono=True)

                x_list, y_list = [x], [y]
                for aug in augments:
                    x_aug, y_aug = data_aug((x, y), aug)
                    x_list.append(x_aug)
                    # y_list.append(y_aug)
                    y_list.append(y)

                for xi, yi in zip(x_list, y_list):
                    if compute_spect:
                        xi_windows = sigwin(xi, window_length * sr, window_type, overlap)
                        yi_windows = sigwin(yi, window_length * sr, window_type, overlap)

                        # print(len(xi_windows))
                        if intra_song_shuffle:
                            np.random.shuffle(xi_windows)
                            np.random.shuffle(yi_windows)

                        for window_index in range(xi_windows.shape[0]):
                            xi_window_spect = np.abs(librosa.stft(xi_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')
                            yi_window_spect = np.abs(librosa.stft(yi_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')

                            if network.lower() == 'u_net':
                                xi_window_spect = xi_window_spect[:-1, :-1]
                                yi_window_spect = yi_window_spect[:-1, :-1]

                            if dB:
                                yi_window_spect = librosa.amplitude_to_db(yi_window_spect, ref=np.max(xi_window_spect))
                                xi_window_spect = librosa.amplitude_to_db(xi_window_spect, ref=np.max)

                                np.clip(xi_window_spect, -80, 0, xi_window_spect)
                                np.clip(yi_window_spect, -80, 0, yi_window_spect)

                            if normalize_from_dataset:
                                if normalize:
                                    xi_window_spect = xi_window_spect / np.max([statistics['maxim_mixture_spect'],
                                                                                np.abs(
                                                                                    statistics['minim_mixture_spect'])])
                                    yi_window_spect = yi_window_spect / np.max([statistics['maxim_' + str(source) +
                                                                                           '_spect'], np.abs(
                                        statistics['minim_' + str(source) + '_spect'])])

                                elif normalize01:
                                    if statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect']:
                                        xi_window_spect = (xi_window_spect - statistics['minim_mixture_spect']) / \
                                                          (statistics['maxim_mixture_spect'] - statistics[
                                                              'minim_mixture_spect'])
                                    else:
                                        xi_window_spect = np.zeros(xi_window_spect.shape)
                                    if statistics['maxim_' + str(source) + '_spect'] - \
                                            statistics['minim_' + str(source) + '_spect']:
                                        yi_window_spect = (yi_window_spect - statistics[
                                            'minim_' + str(source) + '_spect']) / \
                                                          (statistics['maxim_' + str(source) + '_spect'] -
                                                           statistics['minim_' + str(source) + '_spect'])
                                    else:
                                        yi_window_spect = np.zeros(yi_window_spect)

                                elif standardize:
                                    xi_window_spect = (xi_window_spect - statistics['mean_mixture_spect']) / \
                                                      statistics['std_mixture_spect']
                                    yi_window_spect = (yi_window_spect - statistics['mean_' + str(source) + '_spect']) / \
                                                      statistics['std_' + str(source) + '_spect']
                            else:
                                if normalize:
                                    if (not dB and np.max(np.abs(xi_window_spect)) > 10) or (
                                            dB and np.max(np.abs(xi_window_spect)) - np.min(np.abs(xi_window_spect)) > 10):
                                        xi_window_spect = xi_window_spect / np.max(np.abs(xi_window_spect))
                                    else:
                                        xi_window_spect = np.zeros(xi_window_spect.shape)
                                    if (not dB and np.max(np.abs(yi_window_spect)) > 1e4) or (
                                            dB and np.max(np.abs(yi_window_spect)) - np.min(
                                        np.abs(yi_window_spect)) > 10):
                                        yi_window_spect = yi_window_spect / np.max(np.abs(yi_window_spect))
                                    else:
                                        yi_window_spect = np.zeros(yi_window_spect.shape)

                                elif normalize01:
                                    if dB:
                                        xi_window_spect = (xi_window_spect + 80) / 80
                                        yi_window_spect = (yi_window_spect + 80) / 80
                                    else:
                                        if np.max(xi_window_spect) - np.min(xi_window_spect) > 1e-2:
                                            xi_window_spect = (xi_window_spect - np.min(xi_window_spect)) / \
                                                              (np.max(xi_window_spect) - np.min(xi_window_spect))
                                        else:
                                            xi_window_spect = np.zeros(xi_window_spect.shape)
                                        if np.max(yi_window_spect) - np.min(yi_window_spect) > 1e-2:
                                            yi_window_spect = (yi_window_spect - np.min(yi_window_spect)) / \
                                                              (np.max(yi_window_spect) - np.min(yi_window_spect))
                                        else:
                                            yi_window_spect = np.zeros(yi_window_spect.shape)

                                elif standardize:
                                    xi_window_spect = (xi_window_spect - np.mean(xi_window_spect)) / np.std(
                                        xi_window_spect)
                                    yi_window_spect = (yi_window_spect - np.mean(yi_window_spect)) / np.std(
                                        yi_window_spect)

                            #     if window_index == x_windows.shape[0] // 2:
                            #         print(x_window_spect.shape, y_window_spect.shape)
                            #         print('extreme x:', np.max(x_window_spect), np.min(x_window_spect))
                            #         print('extreme y:', np.max(y_window_spect), np.min(y_window_spect))
                            #         print('stats x:', np.mean(x_window_spect), np.std(x_window_spect))
                            #         print('stats y:', np.mean(y_window_spect), np.std(y_window_spect))
                            #           break

                            if subfolder == 'train':
                                train_serialized_spectrograms = serialize_data_1_source(
                                    xi_window_spect.astype(np.float32),
                                    yi_window_spect.astype(np.float32))
                                writer_train.write(train_serialized_spectrograms)
                                train_card += 1
                            elif subfolder == 'val':
                                val_serialized_spectrograms = serialize_data_1_source(
                                    xi_window_spect.astype(np.float32),
                                    yi_window_spect.astype(np.float32))
                                writer_val.write(val_serialized_spectrograms)
                                val_card += 1
                            txt += song + ' ' + str(xi_window_spect.dtype) + ' ' + str(yi_window_spect.dtype) + '\n'


                    else:
                        if normalize_from_dataset:
                            if normalize:
                                xi = xi / np.max([statistics['maxim_mixture'], np.abs(statistics['minim_mixture'])])
                                yi = yi / np.max([statistics['maxim_' + str(source)], np.abs(statistics['minim_'
                                                                                                        + str(
                                    source)])])

                            elif normalize01:
                                xi = (xi - statistics['minim_mixture']) / (statistics['maxim_mixture']
                                                                           - statistics['minim_mixture'])
                                yi = (yi - statistics['minim_' + str(source)]) / (statistics['maxim_' + str(source)] -
                                                                                  statistics['minim_' + str(source)])

                            elif standardize:
                                xi = (xi - statistics['mean_mixture']) / statistics['std_mixture']
                                yi = (yi - statistics['mean_' + str(source)]) / statistics['std_' + str(source)]
                        else:
                            if normalize:
                                xi = xi / np.max(np.abs(xi))
                                yi = yi / np.max(np.abs(yi))

                            elif normalize01:
                                xi = (xi - np.min(xi)) / (np.max(xi) - np.min(xi))
                                yi = (yi - np.min(yi)) / (np.max(yi) - np.min(yi))

                            elif standardize:
                                xi = (xi - np.mean(xi)) / np.std(xi)
                                yi = (yi - np.mean(yi)) / np.std(yi)

                        xi_windows = sigwin(xi, window_length * sr, window_type, overlap)
                        yi_windows = sigwin(yi, window_length * sr, window_type, overlap)

                        if intra_song_shuffle:
                            np.random.shuffle(xi_windows)
                            np.random.shuffle(yi_windows)

                        # print(x_windows.shape, y_windows.shape)
                        # print('extreme x:', np.max(x_windows), np.min(x_windows))
                        # print('extreme y:', np.max(y_windows), np.min(y_windows))
                        # print('stats x:', np.mean(x_windows), np.std(x_windows))
                        # print('stats y:', np.mean(y_windows), np.std(y_windows))
                        # break

                        for window_index in range(xi_windows.shape[0]):
                            if subfolder == 'train':
                                train_serialized_waveforms = serialize_data_1_source(xi_windows[window_index],
                                                                                     yi_windows[window_index])
                                writer_train.write(train_serialized_waveforms)
                                train_card += 1
                            elif subfolder == 'val':
                                val_serialized_waveforms = serialize_data_1_source(xi_windows[window_index],
                                                                                   yi_windows[window_index])
                                writer_val.write(val_serialized_waveforms)
                                val_card += 1

    else:
        raise Exception('Dataset is not correct')

    with open('out.txt', 'w') as f:
        print(txt, file=f)

    write_cardinality(os.path.join('..', 'Cardinality', card_txt), train_card, val_card)



def preprocess_1_source_aug_single_instrument(dataset, resample, sr, window_length, overlap, window_type, compute_spect, dB, n_fft,
                            hop_length, writer_train, writer_val, source, extra_song_shuffle, intra_song_shuffle,
                            normalize_from_dataset, statistics_path, normalize, normalize01, standardize, card_txt,
                            network, augments):
    train_card, val_card = 0, 0
    np.random.seed(42)

    txt = ''
    if (normalize and normalize01) or (normalize01 and standardize) or (normalize and standardize):
        raise Exception('You need to choose only one type of normalization')

    if normalize_from_dataset:
        with open(statistics_path, 'rb') as f:
            statistics = pickle.load(f)

    if dataset.lower() == 'musdb18':
        for subfolder in sorted(os.listdir(os.path.join('..', 'Datasets', 'MUSDB18'))):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            if extra_song_shuffle:
                np.random.shuffle(songs)
            else:
                songs = sorted(songs)

            for i, song in enumerate(songs):
                # if i == 1:
                #     break
                print(i, song)
                if resample:
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)
                else:
                    sr = librosa.get_samplerate(os.path.join(path, song, 'mixture.wav'))
                    bass, _ = librosa.load(os.path.join(path, song, 'bass.wav'), sr=sr, mono=True)
                    drums, _ = librosa.load(os.path.join(path, song, 'drums.wav'), sr=sr, mono=True)
                    vocals, _ = librosa.load(os.path.join(path, song, 'vocals.wav'), sr=sr, mono=True)
                    other, _ = librosa.load(os.path.join(path, song, 'other.wav'), sr=sr, mono=True)

                min_length = np.min((len(vocals), len(bass), len(drums), len(other)))
                vocals = vocals[:min_length]
                bass = bass[:min_length]
                drums = drums[:min_length]
                other = other[:min_length]

                sources = {'vocals': vocals, 'bass': bass, 'drums': drums, 'other': other}

                x_list, y_list = [vocals + bass + drums + other], [sources[source]]
                for aug in augments:
                    x_aug, _ = data_aug((sources[source], sources[source]), aug)
                    x_nonaug = np.zeros(sources[source].shape)
                    for k in sources.keys():
                        if k != source:
                            x_nonaug += sources[k]

                    aug_min_length = np.min((len(x_aug), len(sources[k])))
                    x_list.append(x_aug[:aug_min_length] + x_nonaug[:aug_min_length])
                    y_list.append(sources[source][:aug_min_length])

                for xi, yi in zip(x_list, y_list):
                    if compute_spect:
                        xi_windows = sigwin(xi, window_length * sr, window_type, overlap)
                        yi_windows = sigwin(yi, window_length * sr, window_type, overlap)

                        if intra_song_shuffle:
                            np.random.shuffle(xi_windows)
                            np.random.shuffle(yi_windows)

                        for window_index in range(xi_windows.shape[0]):
                            xi_window_spect = np.abs(librosa.stft(xi_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')
                            yi_window_spect = np.abs(librosa.stft(yi_windows[window_index], n_fft=n_fft,
                                                                  hop_length=hop_length), dtype='float32')

                            if network.lower() == 'u_net':
                                xi_window_spect = xi_window_spect[:-1, :-1]
                                yi_window_spect = yi_window_spect[:-1, :-1]

                            if dB:
                                yi_window_spect = librosa.amplitude_to_db(yi_window_spect, ref=np.max(xi_window_spect))
                                xi_window_spect = librosa.amplitude_to_db(xi_window_spect, ref=np.max)

                                np.clip(xi_window_spect, -80, 0, xi_window_spect)
                                np.clip(yi_window_spect, -80, 0, yi_window_spect)

                            if normalize_from_dataset:
                                if normalize:
                                    xi_window_spect = xi_window_spect / np.max([statistics['maxim_mixture_spect'],
                                                                                np.abs(
                                                                                    statistics['minim_mixture_spect'])])
                                    yi_window_spect = yi_window_spect / np.max([statistics['maxim_' + str(source) +
                                                                                           '_spect'], np.abs(
                                        statistics['minim_' + str(source) + '_spect'])])

                                elif normalize01:
                                    if statistics['maxim_mixture_spect'] - statistics['minim_mixture_spect']:
                                        xi_window_spect = (xi_window_spect - statistics['minim_mixture_spect']) / \
                                                          (statistics['maxim_mixture_spect'] - statistics[
                                                              'minim_mixture_spect'])
                                    else:
                                        xi_window_spect = np.zeros(xi_window_spect.shape)
                                    if statistics['maxim_' + str(source) + '_spect'] - \
                                            statistics['minim_' + str(source) + '_spect']:
                                        yi_window_spect = (yi_window_spect - statistics[
                                            'minim_' + str(source) + '_spect']) / \
                                                          (statistics['maxim_' + str(source) + '_spect'] -
                                                           statistics['minim_' + str(source) + '_spect'])
                                    else:
                                        yi_window_spect = np.zeros(yi_window_spect)

                                elif standardize:
                                    xi_window_spect = (xi_window_spect - statistics['mean_mixture_spect']) / \
                                                      statistics['std_mixture_spect']
                                    yi_window_spect = (yi_window_spect - statistics['mean_' + str(source) + '_spect']) / \
                                                      statistics['std_' + str(source) + '_spect']
                            else:
                                if normalize:
                                    if (not dB and np.max(np.abs(xi_window_spect)) > 10) or (
                                            dB and np.max(np.abs(xi_window_spect)) - np.min(np.abs(xi_window_spect)) > 10):
                                        xi_window_spect = xi_window_spect / np.max(np.abs(xi_window_spect))
                                    else:
                                        xi_window_spect = np.zeros(xi_window_spect.shape)
                                    if (not dB and np.max(np.abs(yi_window_spect)) > 1e4) or (
                                            dB and np.max(np.abs(yi_window_spect)) - np.min(
                                        np.abs(yi_window_spect)) > 10):
                                        yi_window_spect = yi_window_spect / np.max(np.abs(yi_window_spect))
                                    else:
                                        yi_window_spect = np.zeros(yi_window_spect.shape)

                                elif normalize01:
                                    if dB:
                                        xi_window_spect = (xi_window_spect + 80) / 80
                                        yi_window_spect = (yi_window_spect + 80) / 80
                                    else:
                                        if np.max(xi_window_spect) - np.min(xi_window_spect) > 1e-2:
                                            xi_window_spect = (xi_window_spect - np.min(xi_window_spect)) / \
                                                              (np.max(xi_window_spect) - np.min(xi_window_spect))
                                        else:
                                            xi_window_spect = np.zeros(xi_window_spect.shape)
                                        if np.max(yi_window_spect) - np.min(yi_window_spect) > 1e-2:
                                            yi_window_spect = (yi_window_spect - np.min(yi_window_spect)) / \
                                                              (np.max(yi_window_spect) - np.min(yi_window_spect))
                                        else:
                                            yi_window_spect = np.zeros(yi_window_spect.shape)

                                elif standardize:
                                    xi_window_spect = (xi_window_spect - np.mean(xi_window_spect)) / np.std(
                                        xi_window_spect)
                                    yi_window_spect = (yi_window_spect - np.mean(yi_window_spect)) / np.std(
                                        yi_window_spect)

                            #     if window_index == x_windows.shape[0] // 2:
                            #         print(x_window_spect.shape, y_window_spect.shape)
                            #         print('extreme x:', np.max(x_window_spect), np.min(x_window_spect))
                            #         print('extreme y:', np.max(y_window_spect), np.min(y_window_spect))
                            #         print('stats x:', np.mean(x_window_spect), np.std(x_window_spect))
                            #         print('stats y:', np.mean(y_window_spect), np.std(y_window_spect))
                            #           break

                            if subfolder == 'train':
                                train_serialized_spectrograms = serialize_data_1_source(
                                    xi_window_spect.astype(np.float32),
                                    yi_window_spect.astype(np.float32))
                                writer_train.write(train_serialized_spectrograms)
                                train_card += 1
                            elif subfolder == 'val':
                                val_serialized_spectrograms = serialize_data_1_source(
                                    xi_window_spect.astype(np.float32),
                                    yi_window_spect.astype(np.float32))
                                writer_val.write(val_serialized_spectrograms)
                                val_card += 1
                            txt += song + ' ' + str(xi_window_spect.dtype) + ' ' + str(yi_window_spect.dtype) + '\n'


                    else:
                        if normalize_from_dataset:
                            if normalize:
                                xi = xi / np.max([statistics['maxim_mixture'], np.abs(statistics['minim_mixture'])])
                                yi = yi / np.max([statistics['maxim_' + str(source)], np.abs(statistics['minim_'
                                                                                                        + str(
                                    source)])])

                            elif normalize01:
                                xi = (xi - statistics['minim_mixture']) / (statistics['maxim_mixture']
                                                                           - statistics['minim_mixture'])
                                yi = (yi - statistics['minim_' + str(source)]) / (statistics['maxim_' + str(source)] -
                                                                                  statistics['minim_' + str(source)])

                            elif standardize:
                                xi = (xi - statistics['mean_mixture']) / statistics['std_mixture']
                                yi = (yi - statistics['mean_' + str(source)]) / statistics['std_' + str(source)]
                        else:
                            if normalize:
                                xi = xi / np.max(np.abs(xi))
                                yi = yi / np.max(np.abs(yi))

                            elif normalize01:
                                xi = (xi - np.min(xi)) / (np.max(xi) - np.min(xi))
                                yi = (yi - np.min(yi)) / (np.max(yi) - np.min(yi))

                            elif standardize:
                                xi = (xi - np.mean(xi)) / np.std(xi)
                                yi = (yi - np.mean(yi)) / np.std(yi)

                        xi_windows = sigwin(xi, window_length * sr, window_type, overlap)
                        yi_windows = sigwin(yi, window_length * sr, window_type, overlap)

                        if intra_song_shuffle:
                            np.random.shuffle(xi_windows)
                            np.random.shuffle(yi_windows)

                        # print(x_windows.shape, y_windows.shape)
                        # print('extreme x:', np.max(x_windows), np.min(x_windows))
                        # print('extreme y:', np.max(y_windows), np.min(y_windows))
                        # print('stats x:', np.mean(x_windows), np.std(x_windows))
                        # print('stats y:', np.mean(y_windows), np.std(y_windows))
                        # break

                        for window_index in range(xi_windows.shape[0]):
                            if subfolder == 'train':
                                train_serialized_waveforms = serialize_data_1_source(xi_windows[window_index],
                                                                                     yi_windows[window_index])
                                writer_train.write(train_serialized_waveforms)
                                train_card += 1
                            elif subfolder == 'val':
                                val_serialized_waveforms = serialize_data_1_source(xi_windows[window_index],
                                                                                   yi_windows[window_index])
                                writer_val.write(val_serialized_waveforms)
                                val_card += 1

    else:
        raise Exception('Dataset is not correct')

    with open('out.txt', 'w') as f:
        print(txt, file=f)

    write_cardinality(os.path.join('..', 'Cardinality', card_txt), train_card, val_card)
