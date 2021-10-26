from imports import *


def sigwin(x, window_length, w_type, overlap):
    """
    - w_type[string] can be: -rect
                             -boxcar
                             -triang
                             -blackman
                             -hamming
                             -hann
                             -bartlett
                             -flattop
                             -parzen
                             -bohman
                             -blackmanharris
                             -nuttall
                             -barthann

    - overlap [percentage]
    - l[sample number]
    - x[list or np.array]
    """

    overlap = overlap / 100
    w = []
    delay = int((1 - overlap) * window_length)

    if w_type != 'rect':
        win = windows.get_window(w_type, window_length)

    for i in range(0, len(x), delay):
        if i + window_length <= len(x):
            if w_type == 'rect':
                w.append(x[i:i + window_length])
            else:
                w.append(np.multiply(win, x[i:i + window_length]))

    return np.asanyarray(w, dtype='float32')


def spectrogram_windowing(spect, window_length, overlap):
    """
    :param spect: Spectrogram to be windowed
    :param window_length: The length of the spectrogram window (in time)
    :param overlap: The overlapping percent of the windows
    :return: An array of spectrogams windowed from the original one
    """
    overlap = overlap / 100
    w = []
    delay = int((1 - overlap) * window_length)

    for i in range(0, spect.shape[1], delay):
        if i + window_length <= spect.shape[1]:
            w.append(spect[:, i:i + window_length])

    return np.asanyarray(w)


def sigrec(w_signal, overlap, mode='MEAN'):
    """
    Arguments:
        - w_signal: an array with the windows of size #windows x window_length
        - overlap: the percentage of overlapping between windows
        - mode: method to reconstruct the signal:
            - 'OLA' for overlap and addition
            - 'MEAN' for overlap and mean (default if not 'OLA')

    Outputs:
        - x: the reconstructed signal of size signal_length
    """

    n = len(w_signal)  # number of windows
    overlap = overlap / 100  # calc percentage
    window_length = len(w_signal[0])  # window len

    non_ov = int((1 - overlap) * window_length)  # non overlapping section of 2 windows
    lenx = (n - 1) * non_ov + window_length  # len of signal to reconstruct. formula might be wrong.
    print('Reconstructed signal shape: ', lenx)
    delay = non_ov  # used to delay i'th window when creating the matrix that will be averaged

    w_frm_aux = np.zeros((n, lenx), dtype='float32')  # size = windows x signal_length
    for i in range(n):
        crt = np.zeros(i * delay).tolist()
        crt.extend(w_signal[i])
        crt.extend(np.zeros(lenx - i * delay - window_length).tolist())

        w_frm_aux[i] += crt

    summ = np.sum(w_frm_aux, axis=0)
    if mode == 'OLA':
        return summ

    nonzero = w_frm_aux != 0
    divvect = np.sum(nonzero, axis=0)
    divvect[divvect == 0] = 1  # avoid division by zero
    x = summ / divvect

    return np.asanyarray(x)


def write_cardinality(path, train_card, val_card):
    """
    Write the cardinality of a TFRecordDataset in a ".txt" file.

    Arguments:
        - path [string], relative path to the ".txt" file
        - train_card [int], the length of the train TFRecordDataset
        - val_card [int], the length of the validation TFRecordDataset

    Raises:
        - TypeError if cardinality is not an integer
    """

    if type(train_card) != int or type(val_card) != int:
        raise TypeError("Cardinality is not an integer")
    else:
        dir_path = os.path.join(*path.split(os.sep)[0:-1])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        f = open(path, 'w')
        f.write(str(train_card) + ' ' + str(val_card))
        f.close()


def load_cardinality(path):
    """
    Arguments:
        - path [string], relative path to the ".txt" file

    Output:
        - cardinality [int], the length of the TFRecordDataset
    """

    f = open(path, 'r')
    cardinality = f.read()
    f.close()
    return int(cardinality)


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))


def serialize_data_1_source(mixture, source):
    feat_dict = {
                'mixture': bytes_feature(mixture),
                'source': bytes_feature(source)
    }

    ftexample = tf.train.Example(features=tf.train.Features(feature=feat_dict))
    ftserialized = ftexample.SerializeToString()
    return ftserialized


def serialize_data_all_sources(mixture, bass, drums, vocals, other):
    feat_dict = {
                'mixture': bytes_feature(mixture),
                'bass': bytes_feature(bass),
                'drums': bytes_feature(drums),
                'vocals': bytes_feature(vocals),
                'other': bytes_feature(other)
    }

    ftexample = tf.train.Example(features=tf.train.Features(feature=feat_dict))
    ftserialized = ftexample.SerializeToString()
    return ftserialized


def parse_and_decode_1_source(example_proto):
    feature_description = {
        'mixture': tf.io.FixedLenFeature([], tf.string),
        'source': tf.io.FixedLenFeature([], tf.string),
    }
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_mixture = tf.io.parse_tensor(element['mixture'], 'float32')
    decoded_source = tf.io.parse_tensor(element['source'], 'float32')

    # decoded_mixture = tf.expand_dims(decoded_mixture, axis=-1)
    # decoded_source = tf.expand_dims(decoded_source, axis=-1)

    return decoded_mixture, decoded_source


def parse_and_decode_all_sources(example_proto):
    feature_description = {
        'mixture': tf.io.FixedLenFeature([], tf.string),
        'bass': tf.io.FixedLenFeature([], tf.string),
        'drums': tf.io.FixedLenFeature([], tf.string),
        'vocals': tf.io.FixedLenFeature([], tf.string),
        'other': tf.io.FixedLenFeature([], tf.string),
    }
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_mixture = tf.io.parse_tensor(element['mixture'], 'float32')
    decoded_bass = tf.io.parse_tensor(element['bass'], 'float32')
    decoded_drums = tf.io.parse_tensor(element['drums'], 'float32')
    decoded_vocals = tf.io.parse_tensor(element['vocals'], 'float32')
    decoded_other = tf.io.parse_tensor(element['other'], 'float32')

    # decoded_mixture = tf.expand_dims(decoded_mixture, axis=-1)
    # decoded_bass = tf.expand_dims(decoded_bass, axis=-1)
    # decoded_drums = tf.expand_dims(decoded_drums, axis=-1)
    # decoded_vocals = tf.expand_dims(decoded_vocals, axis=-1)
    # decoded_other = tf.expand_dims(decoded_other, axis=-1)

    return decoded_mixture, decoded_bass, decoded_drums, decoded_vocals, decoded_other
