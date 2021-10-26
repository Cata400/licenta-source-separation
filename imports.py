import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, Cropping1D, Reshape, \
    Bidirectional, LSTM, Concatenate, Multiply, Cropping2D, Lambda
from tensorflow.keras.optimizers import SGD
import datetime
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import librosa
from sklearn.utils import shuffle
import random
import pickle
import librosa.display
import scipy
from pydub import AudioSegment
import ntpath