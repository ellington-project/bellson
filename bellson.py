#!/usr/bin/env python3
import logging

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from lib.ellington_library import EllingtonLibrary, Track
from lib.audio import AudioTrack
from lib.generator import DataGenerator



def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    el = EllingtonLibrary.from_file("data/example.el")


    # Define some learning stuff...
    class_names = [str(n) for n in range(0, 400)]

    model = keras.Sequential([
        keras.layers.Dense(400, input_shape=(10000,)),
        keras.layers.Dense(400, activation=tf.nn.relu), 
        keras.layers.Dense(400, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    count = str(len(el.tracks))
    
    for i in range(0, 10): 
        ix = 0
        for t in el.tracks: 
            ix = ix + 1
            print("Track: " + str(t.trackname) + " " + str(ix) + " / " + count)
            print("\t Testing audio data: ")
            
            audiotrack = AudioTrack(t)

            print("\t Training audio data: ")

            label = np.array([t.bpm])

            audiotrack.load()
            audiotrack.save_spectrogram()

            # for ad in audiotrack.spect_intervals():
            #     logging.debug("Audio data recieved")
            #     logging.debug("Audio of shape: " + str(ad[2].shape))

            #     data = np.array(ad[2], ndmin=2)
                

            #     logging.debug("Data shape" + str(data.shape))
            #     logging.debug("Labels shape" + str(label.shape))

            #     loss = model.train_on_batch(x=data, y=label)
            #     logging.info("Model loss = " + str(loss))

            # print("\t Testing audio data:")
            # for ad in audiotrack.audio_intervals(True): 
            #     logging.debug("Audio data recieved")
            #     logging.debug("Audio of shape: " + str(ad[2].shape))

            #     data = np.array(ad[2], ndmin=2)

            #     preds = np.argmax(model.predict_on_batch(x=data))

            #     logging.info("Model preds from test: = " + str(preds) + ", actual : " + str(t.bpm))

            # print(ad[0:10])



if __name__ == '__main__':
    main()