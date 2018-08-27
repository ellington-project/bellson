#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib


import numpy as np

from trainer.ellington_library import EllingtonLibrary, Track
from trainer.audio import Audio
from trainer.generator import LibraryIterator, TrackIterator
from trainer.model import model_gen

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, jobd):
        self.step = 0
        self.jobd = jobd
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step = self.step + 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        metrics_log = ''
        for (k, v) in self.metric_cache.items():
            if abs(v) > 1e-3:
                metrics_log += ' - %s: %.4f' % (k, v)
            else:
                metrics_log += ' - %s: %.4e' % (k, v)

        print('step: {}/{} ::{}'.format(self.step,
                                        self.params['steps'],
                                        metrics_log))
        self.metric_cache.clear()

        gc.collect()
            
            

    def on_epoch_end(self, epoch, logs={}): 
        print("Saving model")

        # Save the model locally
        self.model.save('model.h5')

        # Save the model to the Cloud Storage bucket's jobs directory
        print("Saving to : " + self.jobd)
        with file_io.FileIO('model.h5', mode='rb') as input_f:
            with file_io.FileIO(self.jobd + '-model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())





def main(data_dir="data/smnp/", ellington_lib="data/example.el", job_dir="logs"):
    # Start logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)

    # List the available tensorflow devices
    logging.info(device_lib.list_local_devices())        

    # Set up the data input etc.
    train_lib = EllingtonLibrary.from_file(ellington_lib)
    valid_lib = EllingtonLibrary.from_file(ellington_lib)

    # Set up the generators to yield training data
    training_gen = LibraryIterator(
        train_lib, data_dir, samples=128, batchsize=512, start=30, end=150, iterations=1)
    validation_gen = LibraryIterator(
        valid_lib, data_dir, samples=4, batchsize=64, start=30, end=200, iterations=1)

    # Fix an input size for our model
    input_time_dim = 1720
    input_freq_dim = 256

    # Create the model, print info
    model = model_gen(input_time_dim, input_freq_dim)
    print(model.summary())

    # Compile the model
    sgd = keras.optimizers.SGD(
        lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=['mae', 'msle', 'mape'])

    # Set up callbacks - one for tensorboard
    tfcallback = keras.callbacks.TensorBoard(log_dir=job_dir + "/tensorboard/,
                                           histogram_freq=0,
                                           write_grads=True,
                                           write_graph=False,
                                           write_images=False,
                                           batch_size=32)
    # And another for our custom callback that saves the model.
    bcallback = CustomCallback(job_dir)

    # One for a progress bar
    prog_bar = keras.callbacks.ProgbarLogger(count_mode='steps')

    # Fit the model using all of the above!
    model.fit_generator(
        generator = training_gen.batch(), 
        steps_per_epoch = training_gen.len(), 
        epochs = 1000, 
        verbose = 2, 
        callbacks = [tfcallback, bcallback, prog_bar], 
        validation_data = validation_gen.batch(), 
        validation_steps = validation_gen.len(),
        use_multiprocessing=True 
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to training data, in the form of compressed numpy arrays')
    parser.add_argument('--ellington-lib', required=True, help='The ellington library from which to read track names and BPMs')
    parser.add_argument('--job-dir', required=True, help='The directory to export the model, and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
