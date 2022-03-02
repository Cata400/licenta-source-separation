import numpy as np
import tensorflow as tf
import os
from utils import *
import librosa.display


# name = 'train_small.tfrecord'
# tf_record_path = os.path.join('..', 'TFRecords')
# train_path = os.path.join(tf_record_path, name)
#
# train_dataset = tf.data.TFRecordDataset(train_path)
# train = train_dataset.map(parse_and_decode_1_source)
#
# mean_spect = 0
# for i, data in enumerate(train.take(-1)):
#     print(i)
#
#     mixture = data[0].numpy()
#     mean_spect += mixture
#
# mean_spect /= (i + 1)
#
# fig = plt.figure()
# img1 = librosa.display.specshow(mean_spect, y_axis='log', sr=44100, hop_length=512, x_axis='time')
# fig.colorbar(img1)
# plt.show()
#
# mean = np.mean(mean_spect, axis=-1)
# std = np.std(mean_spect, axis=-1)
# print(mean.shape)
#
#
# print(mean_spect[20, :])
# print(mean[20])
# print(std[20])
#
# np.save(os.path.join('..', 'Models', 'open_unmix_mean.npy'), mean)
# np.save(os.path.join('..', 'Models', 'open_unmix_std.npy'), std)
#
# mean = np.load(os.path.join('..', 'Models', 'open_unmix_mean.npy'))
# std = np.load(os.path.join('..', 'Models', 'open_unmix_std.npy'))
#
# print(np.min(mean), np.max(mean), np.min(std), np.max(std))

name = 'train_small.tfrecord'
tf_record_path = os.path.join('..', 'TFRecords')
train_path = os.path.join(tf_record_path, name)

train_dataset = tf.data.TFRecordDataset(train_path)
train = train_dataset.map(parse_and_decode_1_source)

mean_list = []
std_list = []
for i, data in enumerate(train.take(-1)):
    print(i)
    mixture = data[0].numpy()

    mean = np.mean(mixture, axis=-1)
    std = np.std(mixture, axis=-1)

    mean_list.append(mean)
    std_list.append(std)

mean_list = np.asanyarray(mean_list)
std_list = np.asanyarray(std_list)
print(mean_list.shape)

global_mean = np.mean(mean_list, axis=0)
global_std = np.sqrt(np.mean((mean_list - global_mean) ** 2 + std_list ** 2, axis=0))
print(global_mean.shape)
print(global_std.shape)


np.save(os.path.join('..', 'Models', 'open_unmix_mean2.npy'), global_mean)
np.save(os.path.join('..', 'Models', 'open_unmix_std2.npy'), global_std)

mean = np.load(os.path.join('..', 'Models', 'open_unmix_mean.npy'))
std = np.load(os.path.join('..', 'Models', 'open_unmix_std.npy'))

print(np.min(mean), np.max(mean), np.min(std), np.max(std))
print(np.min(global_mean), np.max(global_mean), np.min(global_std), np.max(global_std))
print(np.allclose(mean, global_mean), np.allclose(std, global_std))

