#!/usr/bin/env python3
import logging

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from lib.ellington_library import EllingtonLibrary, Track
from lib.audio import Audio
from lib.generator import LibraryIterator, TrackIterator


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    el = EllingtonLibrary.from_file("data/example.el")

    # Define some learning stuff...
    class_names = [str(n) for n in range(0, 400)]

    model = keras.Sequential([
        keras.layers.Conv2D(32, (21,21), strides=(3,3), activation='relu',
                            input_shape=(1025, 861, 1)),

        keras.layers.Conv2D(32, (11,11), strides=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        # keras.layers.Conv2D(64, (5, 5), activation='relu'),
        # keras.layers.Conv2D(64, (5, 5), activation='relu'),
        # keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(), 
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(400, activation=tf.nn.softmax)
    ])

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    count = str(len(el.tracks))

    libgen = LibraryIterator(el)

    for s in libgen.batch(): 
        print("Expected " + str(libgen.len()) + " samples")
        print(str(s))

    # model.fit_generator(libgen.iter)

    # for i in range(0, 10):
    #     ix = 0
    #     for t in el.tracks:
    #         ix = ix + 1
    #         print("Track: " + str(t.trackname) + " " + str(ix) + " / " + count)
    #         print("\t Testing audio data: ")

    #         audiotrack = Audio(t)

    #         print("\t Training audio data: ")

    #         label = np.array([t.bpm])

    #         audiotrack.load()
    #         audiotrack.save_spectrogram()

    #         print("\t Training data:")
    #         for ad in audiotrack.spect_intervals():
    #             logging.debug("Audio data recieved")
    #             logging.debug("Audio of shape: " + str(ad[2].shape))

    #             (w, h) = ad[2].shape
    #             data = np.reshape(ad[2], (1, w, h, 1))

    #             logging.debug("Data shape" + str(data.shape))
    #             logging.debug("Labels shape" + str(label.shape))

    #             loss = model.train_on_batch(x=data, y=label)
    #             logging.info("Model loss = " + str(loss))

    #         print("\t Testing data:")
    #         for ad in audiotrack.spect_intervals(True):
    #             logging.debug("Audio data recieved")
    #             logging.debug("Audio of shape: " + str(ad[2].shape))

    #             (w, h) = ad[2].shape
    #             data = np.reshape(ad[2], (1, w, h, 1))

    #             preds = np.argmax(model.predict_on_batch(x=data))

    #             logging.info("Model preds from test: = " +
    #                          str(preds) + ", actual : " + str(t.bpm))


if __name__ == '__main__':
    main()
