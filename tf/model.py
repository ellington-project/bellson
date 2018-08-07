#!/usr/bin/env python3
import tensorflow as tf 
from tensorflow import keras

# sess = tf.Session()

# from keras import backend as K
# K.set_session(sess)

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(784,)),
    keras.layers.Activation('relu'),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax'),
])