import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, Cropping1D, Reshape, \
    Bidirectional, LSTM, Concatenate, Multiply, Cropping2D, Lambda, Conv2D, LeakyReLU, Conv2DTranspose, Add, MaxPool2D, \
    UpSampling2D, Flatten, Conv1D
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
from tensorflow.python.framework.ops import disable_eager_execution
from pysndfx import AudioEffectsChain
from torchvision.transforms import Compose
