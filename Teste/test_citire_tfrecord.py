import matplotlib.pyplot as plt
import numpy as np

from utils import *
import librosa.display

import warnings




name = 'spect_musdb18_sr_8k_window_12s_overlap_75_rect_dB_nfft_1024_hop_768_extra_normalize01_vocals_aug'
tf_record_path = os.path.join('..', '..', 'TFRecords')
train_path = os.path.join(tf_record_path, name + '_train.tfrecord')
val_path = os.path.join(tf_record_path, name + '_val.tfrecord')
n_fft = 1024
hop_length = 768
sr = 8192


train_dataset = tf.data.TFRecordDataset(train_path)
train = train_dataset.map(parse_and_decode_1_source)

val_dataset = tf.data.TFRecordDataset(val_path)
val = val_dataset.map(parse_and_decode_1_source)

for i, example in enumerate(train.take(-1)):
    mixture = example[0].numpy()
    source = example[1].numpy()

    mixture = np.squeeze(mixture)
    source = np.squeeze(source)

    print(mixture.shape, source.shape, np.max(mixture), np.max(source), np.min(mixture), np.min(source), mixture.dtype, source.dtype)

    if 10 < i % 100 < 13:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img1 = librosa.display.specshow(mixture, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[0])
            img2 = librosa.display.specshow(source, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
        fig.colorbar(img1, ax=ax)
        ax[0].set_title('Ground Truth vs Prediction')
        # fig.savefig(os.path.join('..', '..', 'Models', 'Spectrograms', 'test.png'))
        plt.show()

# val_dataset = tf.data.TFRecordDataset(val_path)
# val = val_dataset.map(parse_and_decode_1_source)
# val = val.skip(1009)
# for data in val.take(-1):
#     mixture = data[0].numpy()
#     source = data[1].numpy()
#     print(mixture.shape, source.shape)
#     mixture = np.squeeze(mixture)
#     source = np.squeeze(source)
#     fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         img1 = librosa.display.specshow(mixture, y_axis='log', sr=44100, hop_length=512, x_axis='time', ax=ax[0])
#         img2 = librosa.display.specshow(source, y_axis='log', sr=44100, hop_length=512, x_axis='time', ax=ax[1])
#     fig.colorbar(img1, ax=ax)
#     ax[0].set_title('Ground Truth vs Prediction')
#     # fig.savefig(os.path.join('..', '..', 'Models', 'Spectrograms', 'test.png'))
#     plt.show()