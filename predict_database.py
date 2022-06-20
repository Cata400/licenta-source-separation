from Py.custom_losses import L11_norm
from prediction import predict2
from imports import *
from utils import get_name
from custom_callbacks import *
import shutil

dataset = 'MUSDB18'
network = 'u_net'

pred_source = 'other'
pred_aug = False
pred_multiple_sources = False
pred_resample = True
pred_sr = 8192
pred_window_length = 12
pred_overlap = 75
pred_window_type = 'rect'

pred_compute_spect = True
pred_dB = True
pred_n_fft = 1024
pred_hop_length = 768

pred_normalize = False
pred_normalize01 = True
pred_standardize = False
pred_normalize_from_dataset = False

test_path = os.path.join('..', 'Datasets', 'Test')
load_model_name = 'u_net_17_other.h5'
model_path = os.path.join('..', 'Models', load_model_name)
save_song_path = os.path.join('..', 'Predictions', 'Source Phase')
dataset_path = os.path.join('..', 'Datasets', 'MUSDB18')
pred_batch_size = 16

if network.lower() == 'u_net':
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        model = tf.keras.models.load_model(model_path, custom_objects={'L11_norm': L11_norm})

for j, folder in enumerate(sorted(os.listdir(dataset_path))):
    for i, song in enumerate(sorted(os.listdir(os.path.join(dataset_path, folder)))):
        print(folder, i, song)
        # os.mkdir(os.path.join('..', 'Datasets', 'MUSDB18_predict', folder, song))
        predict2(test_path=os.path.join('..', 'Datasets', 'MUSDB18', folder, song, 'mixture.wav'),
                 model_path=model_path,
                 multiple_sources=pred_multiple_sources, compute_spect=pred_compute_spect, resample=pred_resample,
                 sr=pred_sr, window_length=pred_window_length, overlap=pred_overlap, window_type=pred_window_type,
                 dB=pred_dB, n_fft=pred_n_fft, hop_length=pred_hop_length, source=pred_source,
                 normalize_from_dataset=pred_normalize_from_dataset, statistics_path='',
                 normalize=pred_normalize, normalize01=pred_normalize01, standardize=pred_standardize,
                 batch_size=pred_batch_size,
                 save_path=os.path.join(os.path.join('..', 'Datasets', 'MUSDB18_predict', folder, song)),
                 network=network, model=model)

        # os.mkdir(os.path.join('..', 'Datasets', 'MUSDB18_predict', folder, song))
        # shutil.copy2(os.path.join('..', 'Datasets', 'MUSDB18_predict_vocals', folder, song, pred_source + '_pred.wav'),
        #              os.path.join('..', 'Datasets', 'MUSDB18_predict', folder, song, pred_source + '.wav'))


