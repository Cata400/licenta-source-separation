from prediction_source_phase import predict
from imports import *
from utils import get_name
from custom_callbacks import *
from lr_schedulers import *


dataset = 'MUSDB18'
network = 'u_net'

pred_source = 'vocals'
pred_multiple_sources = False
pred_resample = True
pred_sr = 8192
pred_window_length = 12
pred_overlap = 75
pred_window_type = 'rect'

pred_compute_spect = True
pred_n_fft = 1024
pred_hop_length = 768

pred_normalize = False
pred_normalize01 = True
pred_standardize = False
pred_normalize_from_dataset = False

test_path = os.path.join('..', 'Datasets', 'Test')
model_path = os.path.join('..', 'Models')
save_song_path = os.path.join('..', 'Predictions', 'Source Phase')


model_numbers = [1, 3, 15, 17, 19, 26, 27, 32]
dbs = [True, True, True, True, True, True, True, True]
batches = [64, 64, 64, 64, 64, 16, 16, 16]

for nr, pred_dB, pred_batch_size in zip(model_numbers, dbs, batches):
    load_model_name = 'u_net_' + str(nr) + '.h5'
    print('MODEL: ', load_model_name)

    test_song = 'train_1.wav'
    predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
            multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
            sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
            dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
            normalize_from_dataset=pred_normalize_from_dataset, statistics_path='',
            normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
            batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song), network=network)

    test_song = 'val_10.wav'
    predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
            multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
            sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
            dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
            normalize_from_dataset=pred_normalize_from_dataset, statistics_path='',
            normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
            batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song), network=network)

    load_model_name = 'u_net_' + str(nr) + '_best.h5'
    print('MODEL: ', load_model_name)

    test_song = 'train_1.wav'
    predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
            multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
            sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
            dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
            normalize_from_dataset=pred_normalize_from_dataset, statistics_path='',
            normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
            batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song), network=network)

    test_song = 'val_10.wav'
    predict(test_path=os.path.join(test_path, test_song), model_path=os.path.join(model_path, load_model_name),
            multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
            sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
            dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
            normalize_from_dataset=pred_normalize_from_dataset, statistics_path='',
            normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
            batch_size=pred_batch_size, save_path=os.path.join(save_song_path, test_song), network=network)
