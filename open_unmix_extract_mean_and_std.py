from utils import *


name = 'spect_musdb18_sr_44k_window_6s_overlap_75_rect_dB_nfft_4096_hop_1024_extra_normalize01_vocals_train.tfrecord'
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

global_mean = np.mean(mean_list, axis=0)
global_std = np.sqrt(np.mean((mean_list - global_mean) ** 2 + std_list ** 2, axis=0))

np.save(os.path.join('..', 'Models', 'mean_db.npy'), global_mean)
np.save(os.path.join('..', 'Models', 'std_db.npy'), global_std)



