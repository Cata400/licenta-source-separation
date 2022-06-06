import tensorflow.keras.backend as kb
from imports import *
from utils import *


def L11_norm(y_actual, y_pred):
    custom_loss = kb.sum(kb.abs(y_actual - y_pred)) / y_actual.shape[0]
    print('Loss 1', custom_loss * y_actual.shape[0])
    return custom_loss


def L11_norm2(y_actual, y_pred):
    custom_loss = tf.norm(tf.math.abs(y_actual - y_pred), ord=1, axis=None)
    return custom_loss


train_path = os.path.join('..', '..', 'TFRecords',
                          'spect_musdb18_sr_8k_window_12s_overlap_75_rect_dB_nfft_1024_hop_768_extra_normalize01_vocals_train.tfrecord')
train_dataset = tf.data.TFRecordDataset(train_path)
train_dataset = train_dataset.map(parse_and_decode_1_source)
train_dataset = train_dataset.skip(2490)


batch_size = 64
x = []
for example in train_dataset.take(batch_size):
    source = example[1].numpy()
    x.append(source)

x = np.asanyarray(x)
print(source.shape)
print(x.shape)

y = np.random.randn(batch_size, source.shape[0], source.shape[1], 1)
loss = L11_norm(x, y)
loss2 = L11_norm2(x, y)
print(loss)
print(loss2)
