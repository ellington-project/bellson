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

from common.ellington_library import EllingtonLibrary, Track
from trainer.library_iterator import LibraryIterator, TrackIterator
import trainer.model as tmodel
import common.config as config


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, jobd):
        self.jobd = jobd

    def on_train_batch_end(self, batch, logs):
        logging.info(f"Finished train batch: {batch}, logs: {str(logs)}")

    def on_test_batch_end(self, batch, logs):
        logging.info(f"Finished test batch: {batch}, logs: {str(logs)}")

    def on_predict_batch_end(self, batch, logs):
        logging.info(f"Finished predict batch: {batch}, logs: {str(logs)}")

    def on_epoch_end(self, epoch, logs):
        logging.debug("Saving model")

        # Save the model locally
        self.model.save(self.jobd+'/latest-model.h5')

        gc.collect()


def main(cache_dir="/tmp", ellington_lib="data/example.el", job_dir="job"):
    config.cache_directory = cache_dir

    # Set up the data input etc.
    overall_library = EllingtonLibrary.from_file(ellington_lib)
    (train_lib, valid_lib) = overall_library.split_training_validation()

    # Set up the generators to yield training data
    training_gen = LibraryIterator(train_lib, multiplier=1)
    validation_gen = LibraryIterator(valid_lib, multiplier=2)

    logging.info(f"Training length: {train_lib.len()}")
    logging.info(f"Validation length: {valid_lib.len()}")

    validation_dataset = validation_gen.realise_for_validation()

    # Fix an input size for our model
    input_time_dim = 1720
    input_freq_dim = 256

    # Create the model, print info
    model = tmodel.model_gen(input_time_dim, input_freq_dim)
    print(model.summary())

    # Compile the model
    opt = keras.optimizers.SGD(
        lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = keras.optimizers.Adam(learning_rate=1e-2)

    # opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mae', 'msle', 'mape'])

    # Set up callbacks - one for tensorboard
    tfcallback = keras.callbacks.TensorBoard(
        log_dir=job_dir + "/tensorboard/", profile_batch=0)
    # Another for logging data to CSV
    csvlogger = keras.callbacks.CSVLogger(job_dir + "/training.log")

    # Another for saving checkpoints.
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=job_dir +
        "/model-epoch-{epoch:02d}-loss-{val_loss:.8f}.hdf5",
        verbose=1,
        moditor='val_loss',
        mode='min',
        save_weights_only=False,
        save_freq='epoch'
    )

    # And another for our custom callback that logs updates
    bcallback = CustomCallback(job_dir)

    # Fit the model using all of the above!
    model.fit(
        # provide input to x as a generator - don't need to specify y
        x=training_gen,
        # Now many batches of samples per epoch?
        steps_per_epoch=training_gen.batch_count(),
        # Train for at least this many epochs
        epochs=1000,
        # Log a line per epoch
        verbose=2,
        # The callbacks that we wish to run each epoch/batch
        callbacks=[tfcallback, csvlogger,
                   model_checkpoint_callback, bcallback],
        # Our dataset for validating the training of the mode.
        validation_data=validation_dataset,
        # Use a larger queue so that we can get more data in parallel
        # max_queue_size=8,
        # Use a number of workers so that we load data in parallel.
        # workers=2,
        # and use multiprocessing so that we actually use them.
        # use_multiprocessing=True
    )


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True,
                        help='Path to cache directory, for pre-compiled histograms')
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    parser.add_argument('--job-dir', required=True,
                        help='The directory to export the model, and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
