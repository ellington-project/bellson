#!/usr/bin/env python3
import logging
import gc
import objgraph

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from lib.ellington_library import EllingtonLibrary, Track
from lib.audio import Audio
from lib.generator import LibraryIterator, TrackIterator

class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()
            gc.collect()
            # objgraph.show_most_common_types(limit=20)
            # objgraph.get_new_ids()
            

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.ERROR)
    el = EllingtonLibrary.from_file("data/example.el")

    print("Training with library of size: " + str(len(el.tracks)))

    num_filters = 32

    input_time_dim=860
    input_freq_dim=256

    model = keras.Sequential([
        keras.layers.Conv2D(num_filters, (3,1), strides=(1, 1), activation='relu',
                            input_shape=(input_freq_dim, input_time_dim, 1)),
        keras.layers.MaxPool2D(pool_size=(3, 2)),
        keras.layers.Dropout(0.01),

        keras.layers.Conv2D(num_filters*2, (3, 1), strides=(1, 1), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(3, 2)),
        keras.layers.Dropout(0.01),

        keras.layers.Conv2D(num_filters*2*2, (3, 1), strides=(1, 1), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(3, 2)),
        keras.layers.Dropout(0.01),

        # keras.layers.Reshape((input_time_dim, 26 * num_filters)), 

        # keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu')), 
        # keras.layers.TimeDistributed(keras.layers.Dense(8, activation='relu')), 

        keras.layers.Flatten(),

        keras.layers.Dense(400, activation=tf.nn.relu),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(20, activation='sigmoid'),
        keras.layers.Dense(1), 
    ])

    print(model.summary())

    adam = tf.train.AdamOptimizer()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(optimizer=adam,
                  loss='mean_squared_error',
                  metrics=['mae', 'msle', 'mape', 'poisson'])

    count = str(len(el.tracks))

    training_gen = LibraryIterator(el, samples=5, batchsize=10)
    validation_gen = LibraryIterator(el, samples=2, batchsize=4)

    tfcallback = keras.callbacks.TensorBoard(log_dir='./logs',
                                           histogram_freq=0,
                                           write_grads=True,
                                           batch_size=100)

    bcallback = NBatchLogger(1)
    csv_logger = keras.callbacks.CSVLogger('training.log')

    model.fit_generator(training_gen.batch(),
                        steps_per_epoch=training_gen.len(),
                        epochs=5,
                        verbose=2,
                        callbacks=[tfcallback, bcallback, csv_logger],
                        # validation_data=validation_gen.batch(),
                        # validation_steps=validation_gen.len(),
                        class_weight=None,
                        max_queue_size=60,
                        workers=1,
                        use_multiprocessing=True,
                        shuffle=True,
                        initial_epoch=0)

if __name__ == '__main__':
    main()
