import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import parse_and_decode_1_source
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import warnings
import keras.backend as K


class Spectrogram_Callback_1_source(keras.callbacks.Callback):
    def __init__(self, sr, hop_length, val_tfrecord_path, batch_size, model_name):
        self.sr = sr
        self.hop_length = hop_length
        self.val_tfrecord_path = val_tfrecord_path
        self.batch_size = batch_size
        self.model_name = model_name


    def get_test_data(self):
        val_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
        val = val_dataset.map(parse_and_decode_1_source)
        val = val.skip(1009)
        for data in val.take(1):
            mixture = data[0].numpy()
            source = data[1].numpy()

            return mixture, source


    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            mixture, true_source = self.get_test_data()

            mixture = mixture[np.newaxis, :, :]
            true_source = true_source[np.newaxis, :, :]

            predicted_source = self.model.predict(x=mixture, batch_size=self.batch_size)

            true_source = np.squeeze(true_source)
            predicted_source = np.squeeze(predicted_source)

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                true_source_show = librosa.display.specshow(true_source, y_axis='log',
                                                            sr=self.sr, hop_length=self.hop_length, x_axis='time',
                                                            ax=ax[0])

                predicted_source_show = librosa.display.specshow(predicted_source,
                                                                 y_axis='log', sr=self.sr, hop_length=self.hop_length,
                                                                 x_axis='time', ax=ax[1])

            fig.colorbar(true_source_show, ax=ax)#, format="%+2.f dB")
            ax[0].set_title('Ground Truth vs Prediction')
            fig.savefig(os.path.join('..', 'Spectrograms', str(self.model_name) + '_epoch_' + str(epoch) + '.png'))

            print('Saving a validation spectrogram to ../Models/Spectrograms/{}_epoch_{}.png'.format(self.model_name, epoch))
            print('Max and min:', predicted_source.max(), predicted_source.min())

            plt.clf()


class Scale_Layers_Callback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        scale_in = self.model.get_layer(name='ScaleIn')
        scale_out = self.model.get_layer(name='ScaleOut')

        for i, (scale_in_weights, scale_out_weights) in enumerate(zip(scale_in.get_weights(), scale_out.get_weights())):
            np.save(os.path.join('..', 'Models', 'Weights', 'ScaleIn_{}_epoch{}.npy'.format(i, epoch)), scale_in_weights)
            np.save(os.path.join('..', 'Models', 'Weights', 'ScaleOut_{}_epoch{}.npy'.format(i, epoch)), scale_out_weights)


class GetMeanCallback(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if 'lambda' in layer.name:
                inp = self.model.input
                lambda_out = layer.output
                lambda_out = K.eval(layer.input)
                print(lambda_out)

                input_mean = np.mean(lambda_out, axis=-1)
                input_mean = np.expand_dims(input_mean, axis=-1)
                print(type(input_mean))

                input_std = tf.math.reduce_std(lambda_out, axis=-1)
                input_std = tf.expand_dims(input_std, axis=-1)
                input_std = tf.reshape(input_std, (input_std.shape[-2], input_std.shape[-1]))

            if 'ScaleIn' in layer.name:
                layer.set_weights([input_mean, input_std])


class GetEpoch(keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def on_epoch_begin(self, epoch, logs=None):
        with open(os.path.join('..', 'Cardinality', self.model_name + '_epoch.txt'), 'w+') as f:
            f.write(str(epoch))

