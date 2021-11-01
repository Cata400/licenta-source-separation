from utils import *


def parse_dataset(dataset, resample, sr, window_length, overlap, window_type, dB, n_fft, hop_length, file_path, name):
    maxim_dataset, maxim_mixture, maxim_bass, maxim_drums, maxim_vocals, maxim_other = 0, 0, 0, 0, 0, 0
    minim_dataset, minim_mixture, minim_bass, minim_drums, minim_vocals, minim_other = 999, 999, 999, 999, 999, 999

    maxim_spect_dataset, maxim_mixture_spect, maxim_bass_spect, maxim_drums_spect, maxim_vocals_spect, \
    maxim_other_spect = 0, 0, 0, 0, 0, 0
    minim_spect_dataset, minim_mixture_spect, minim_bass_spect, minim_drums_spect, minim_vocals_spect, \
    minim_other_spect = 999, 999, 999, 999, 999, 999

    mean_dataset, mean_mixture_list, mean_bass_list, mean_drums_list, mean_vocals_list, \
    mean_other_list = 0, [], [], [], [], []
    std_dataset, std_mixture_list, std_bass_list, std_drums_list, std_vocals_list, \
    std_other_list = 0, [], [], [], [], []

    mean_spect_dataset, mean_mixture_spect_list, mean_bass_spect_list, mean_drums_spect_list, mean_vocals_spect_list, \
    mean_other_spect_list = 0, [], [], [], [], []
    std_spect_dataset, std_mixture_spect_list, std_bass_spect_list, std_drums_spect_list, std_vocals_spect_list, \
    std_other_spect_list = 0, [], [], [], [], []

    train_card, val_card = 0, 0

    if dataset == 'MUSDB18':
        for subfolder in os.listdir(os.path.join('..', 'Datasets', 'MUSDB18')):
            path = os.path.join('..', 'Datasets', 'MUSDB18', subfolder)
            songs = os.listdir(path)

            for i, song in enumerate(sorted(songs)):
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

                if np.max(mixture) > maxim_mixture:
                    maxim_mixture = np.max(mixture)
                if np.max(bass) > maxim_bass:
                    maxim_bass = np.max(bass)
                if np.max(drums) > maxim_drums:
                    maxim_drums = np.max(drums)
                if np.max(vocals) > maxim_vocals:
                    maxim_vocals = np.max(vocals)
                if np.max(other) > maxim_other:
                    maxim_other = np.max(other)

                if np.min(mixture) < minim_mixture:
                    minim_mixture = np.min(mixture)
                if np.min(bass) < minim_bass:
                    minim_bass = np.min(bass)
                if np.min(drums) < minim_drums:
                    minim_drums = np.min(drums)
                if np.min(vocals) < minim_vocals:
                    minim_vocals = np.min(vocals)
                if np.min(other) < minim_other:
                    minim_other = np.min(other)

                mean_mixture_list.append(np.mean(mixture))
                mean_bass_list.append(np.mean(bass))
                mean_drums_list.append(np.mean(drums))
                mean_vocals_list.append(np.mean(vocals))
                mean_other_list.append(np.mean(other))

                std_mixture_list.append(np.std(mixture))
                std_bass_list.append(np.std(bass))
                std_drums_list.append(np.std(drums))
                std_vocals_list.append(np.std(vocals))
                std_other_list.append(np.std(other))

                mixture_windows = sigwin(mixture, window_length * sr, window_type, overlap)
                bass_windows = sigwin(bass, window_length * sr, window_type, overlap)
                drums_windows = sigwin(drums, window_length * sr, window_type, overlap)
                vocals_windows = sigwin(vocals, window_length * sr, window_type, overlap)
                other_windows = sigwin(other, window_length * sr, window_type, overlap)

                for window_index in range(mixture_windows.shape[0]):
                    if subfolder == 'train':
                        train_card += 1
                    elif subfolder == 'val':
                        val_card += 1

                    mixture_spect = np.abs(librosa.stft(mixture_windows[window_index], n_fft=n_fft,
                                                               hop_length=hop_length))
                    bass_spect = np.abs(librosa.stft(bass_windows[window_index], n_fft=n_fft,
                                                               hop_length=hop_length))
                    drums_spect = np.abs(librosa.stft(drums_windows[window_index], n_fft=n_fft,
                                                               hop_length=hop_length))
                    vocals_spect = np.abs(librosa.stft(vocals_windows[window_index], n_fft=n_fft,
                                                               hop_length=hop_length))
                    other_spect = np.abs(librosa.stft(other_windows[window_index], n_fft=n_fft,
                                                               hop_length=hop_length))

                    if dB:
                        maxim = np.max(mixture_spect)
                        bass_spect = librosa.amplitude_to_db(bass_spect, ref=maxim)
                        drums_spect = librosa.amplitude_to_db(drums_spect, ref=maxim)
                        vocals_spect = librosa.amplitude_to_db(vocals_spect, ref=maxim)
                        other_spect = librosa.amplitude_to_db(other_spect, ref=maxim)
                        mixture_spect = librosa.amplitude_to_db(mixture_spect, ref=maxim)

                        np.clip(mixture_spect, -80, 0, mixture_spect)
                        np.clip(bass_spect, -80, 0, bass_spect)
                        np.clip(drums_spect, -80, 0, drums_spect)
                        np.clip(vocals_spect, -80, 0, vocals_spect)
                        np.clip(other_spect, -80, 0, other_spect)

                    if np.max(mixture_spect) > maxim_mixture_spect:
                        maxim_mixture_spect = np.max(mixture_spect)
                    if np.max(bass_spect) > maxim_bass_spect:
                        maxim_bass_spect = np.max(bass_spect)
                    if np.max(drums_spect) > maxim_drums_spect:
                        maxim_drums_spect = np.max(drums_spect)
                    if np.max(vocals_spect) > maxim_vocals_spect:
                        maxim_vocals_spect = np.max(vocals_spect)
                    if np.max(other_spect) > maxim_other_spect:
                        maxim_other_spect = np.max(other_spect)

                    if np.min(mixture_spect) < minim_mixture_spect:
                        minim_mixture_spect = np.min(mixture_spect)
                    if np.min(bass_spect) < minim_bass_spect:
                        minim_bass_spect = np.min(bass_spect)
                    if np.min(drums_spect) < minim_drums_spect:
                        minim_drums_spect = np.min(drums_spect)
                    if np.min(vocals_spect) < minim_vocals_spect:
                        minim_vocals_spect = np.min(vocals_spect)
                    if np.min(other_spect) < minim_other_spect:
                        minim_other_spect = np.min(other_spect)

                    mean_mixture_spect_list.append(np.mean(mixture_spect))
                    mean_bass_spect_list.append(np.mean(bass_spect))
                    mean_drums_spect_list.append(np.mean(drums_spect))
                    mean_vocals_spect_list.append(np.mean(vocals_spect))
                    mean_other_spect_list.append(np.mean(other_spect))

                    std_mixture_spect_list.append(np.std(mixture_spect))
                    std_bass_spect_list.append(np.std(bass_spect))
                    std_drums_spect_list.append(np.std(drums_spect))
                    std_vocals_spect_list.append(np.std(vocals_spect))
                    std_other_spect_list.append(np.std(other_spect))

        maxim_dataset = np.max(np.asanyarray([maxim_mixture, maxim_bass, maxim_drums, maxim_vocals, maxim_other]))
        minim_dataset = np.min(np.asanyarray([minim_mixture, minim_bass, minim_drums, minim_vocals, minim_other]))

        maxim_spect_dataset = np.max(np.asanyarray([maxim_mixture_spect, maxim_bass_spect, maxim_drums_spect,
                                                    maxim_vocals_spect, maxim_other_spect]))
        minim_spect_dataset = np.min(np.asanyarray([minim_mixture_spect, minim_bass_spect, minim_drums_spect,
                                                    minim_vocals_spect, minim_other_spect]))

        mean_mixture = np.mean(np.asanyarray(mean_mixture_list))
        mean_bass = np.mean(np.asanyarray(mean_bass_list))
        mean_drums = np.mean(np.asanyarray(mean_drums_list))
        mean_vocals = np.mean(np.asanyarray(mean_vocals_list))
        mean_other = np.mean(np.asanyarray(mean_other_list))
        mean_dataset = np.mean(np.asanyarray([mean_mixture, mean_bass, mean_drums, mean_vocals, mean_other]))

        std_mixture = np.sqrt(np.mean((mean_mixture_list - mean_mixture) ** 2 + np.asanyarray(std_mixture_list) ** 2))
        std_bass = np.sqrt(np.mean((mean_bass_list - mean_bass) ** 2 + np.asanyarray(std_bass_list) ** 2))
        std_drums = np.sqrt(np.mean((mean_drums_list - mean_drums) ** 2 + np.asanyarray(std_drums_list) ** 2))
        std_vocals = np.sqrt(np.mean((mean_vocals_list - mean_vocals) ** 2 + np.asanyarray(std_vocals_list) ** 2))
        std_other = np.sqrt(np.mean((mean_other_list - mean_other) ** 2 + np.asanyarray(std_other_list) ** 2))
        std_dataset = np.sqrt(np.mean(np.mean(np.asanyarray([mean_mixture, mean_bass, mean_drums, mean_vocals,
                    mean_other]) - mean_dataset) ** 2 + np.asanyarray(np.asanyarray([std_mixture, std_bass,
                                                                        std_drums, std_vocals, std_other]))))

        mean_mixture_spect = np.mean(np.asanyarray(mean_mixture_spect_list))
        mean_bass_spect = np.mean(np.asanyarray(mean_bass_spect_list))
        mean_drums_spect = np.mean(np.asanyarray(mean_drums_spect_list))
        mean_vocals_spect = np.mean(np.asanyarray(mean_vocals_spect_list))
        mean_other_spect = np.mean(np.asanyarray(mean_other_spect_list))
        mean_spect_dataset = np.mean(np.asanyarray([mean_mixture_spect, mean_bass_spect, mean_drums_spect,
                                                    mean_vocals_spect, mean_other_spect]))

        std_spect_mixture = np.sqrt(np.mean((mean_mixture_spect_list - mean_mixture_spect) ** 2 +
                                            np.asanyarray(std_mixture_spect_list) ** 2))
        std_spect_bass = np.sqrt(np.mean((mean_bass_spect_list - mean_bass_spect) ** 2
                                         + np.asanyarray(std_bass_spect_list) ** 2))
        std_spect_drums = np.sqrt(np.mean((mean_drums_spect_list - mean_drums_spect) ** 2
                                          + np.asanyarray(std_drums_spect_list) ** 2))
        std_spect_vocals = np.sqrt(np.mean((mean_vocals_spect_list - mean_vocals_spect) ** 2
                                           + np.asanyarray(std_vocals_spect_list) ** 2))
        std_spect_other = np.sqrt(np.mean((mean_other_spect_list - mean_other_spect) ** 2
                                          + np.asanyarray(std_other_spect_list) ** 2))
        std_dataset_spect = np.sqrt(np.mean(np.mean(np.asanyarray([mean_mixture_spect, mean_bass_spect,
                         mean_drums_spect, mean_vocals_spect, mean_other_spect]) - mean_spect_dataset) ** 2 +
                        np.asanyarray(np.asanyarray([std_spect_mixture, std_spect_bass, std_spect_drums,
                                                     std_spect_vocals, std_spect_other]))))

    statistics = {'maxim_dataset': maxim_dataset, 'maxim_mixture': maxim_mixture, 'maxim_bass': maxim_bass, 'maxim_drums': maxim_drums,
              'maxim_vocals': maxim_vocals, 'maxim_other': maxim_other, 'minim_dataset': minim_dataset,
              'minim_mixture': minim_mixture, 'minim_bass': minim_bass, 'minim_drums': minim_drums, 'minim_vocals': minim_vocals,
              'minim_other': minim_other, 'maxim_dataset_spect': maxim_spect_dataset,
              'maxim_mixture_spect': maxim_mixture_spect, 'maxim_bass_spect': maxim_bass_spect, 'maxim_drums_spect': maxim_drums_spect,
              'maxim_vocals_spect': maxim_vocals_spect, 'maxim_other_spect': maxim_other_spect,
              'minim_dataset_spect': minim_spect_dataset, 'minim_mixture_spect': minim_mixture_spect,
              'minim_bass_spect': minim_bass_spect, 'minim_drums_spect': minim_drums_spect, 'minim_vocals_spect': minim_vocals_spect,
              'minim_other_spect': minim_other_spect, 'mean_dataset': mean_dataset, 'mean_mixture': mean_mixture,
              'mean_bass': mean_bass, 'mean_drums': mean_drums, 'mean_vocals': mean_vocals,
              'mean_other': mean_other, 'std_dataset': std_dataset, 'std_mixture': std_mixture,
              'std_bass': std_bass, 'std_drums': std_drums, 'std_vocals': std_vocals, 'std_other': std_other,
              'mean_dataset_spect': mean_spect_dataset, 'mean_mixture_spect': mean_mixture_spect,
              'mean_bass_spect': mean_bass_spect, 'mean_drums_spect': mean_drums_spect,
              'mean_vocals_spect': mean_vocals_spect, 'mean_other_spect': mean_other_spect,
              'std_dataset_spect': std_spect_dataset, 'std_mixture_spect': std_spect_mixture,
              'std_bass_spect': std_spect_bass, 'std_drums_spect': std_spect_drums,
              'std_vocals_spect': std_spect_vocals, 'std_other_spect': std_spect_other, 'train_card': train_card,
              'val_card': val_card}

    pic = open(os.path.join(file_path, name + '_statistics.pkl'), "wb")
    pickle.dump(statistics, pic)
    pic.close()


dataset = 'MUSDB18'
resample = False
sr = 16000
window_length = 1
overlap = 50
window_type = 'rect'
n_fft = 2048
hop_length = 512
file_path = '../Cardinality'
name = 'musdb18_sr_44k_window_1s_overlap_50_rect_nfft_2048_hop_512'
parse_dataset(dataset, resample, sr, window_length, overlap, window_type, n_fft, hop_length, file_path, name)
with open(os.path.join(file_path, name + '_statistics.pkl'), 'rb') as f:
    print(pickle.load(f))

