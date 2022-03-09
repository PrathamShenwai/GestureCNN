import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
GESTURE_CATEGORIES = 3
base_model = Sequential()
base_model.add(SqueezeNet(input_shape=(225, 225, 3), include_top=False))
base_model.add(Dropout(0.5))
base_model.add(Convolution2D(GESTURE_CATEGORIES, (1, 1), padding='valid'))
base_model.add(Activation('relu'))
base_model.add(GlobalAveragePooling2D())
base_model.add(Activation('softmax'))
base_model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)