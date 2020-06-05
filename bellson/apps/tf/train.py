#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib

import time
import psutil
import sys

import numpy as np

from ...libbellson.ellington_library import EllingtonLibrary, Track
from ...libbellson.library_iterator import LibraryIterator, TrackIterator
from ...libbellson import model as tmodel
from ...libbellson import config


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, jobd, training_batches, validation_batches):
        self.jobd = jobd
        self.current_epoch = 0
        self.best_val = 1e10
        self.best = 1e10
        self.training_batches = training_batches
        self.validation_batches = validation_batches
        self.batch_begin_times = {}
        self.epoch_begin_time = None

    def format_batch_logs(self, logs):
        # Get the batch loss
        try:
            loss_str = f"loss: {logs['loss']:.8f}, "
        except KeyError:
            loss_str = ""

        try:
            mse_str = f"mse: {logs['mse']:.8f}, "
        except KeyError:
            mse_str = ""

        # Get the batch mae
        try:
            mae_str = f"mae: {logs['mae']:.7f}, "
        except KeyError:
            mae_str = ""

        # Get the batch msle
        try:
            msle_str = f"msle: {logs['msle']:.7f}, "
        except KeyError:
            msle_str = ""

        # Get the batch mape
        try:
            mape_str = f"mape: {logs['mape']:.5f}, "
        except KeyError:
            mape_str = ""

        # Get the batch size
        try:
            size_str = f"size: {logs['size']}, "
        except KeyError:
            size_str = ""

        gb = 1024 * 1024 * 1024

        mem_avail = psutil.virtual_memory().available / gb
        mem_used = psutil.virtual_memory().used / gb
        mem_str = f"mem avail/used: {mem_avail:.3f}/{mem_used:.3f}"

        return loss_str + mse_str + mae_str + msle_str + mape_str + size_str + mem_str

    def format_time(self, time):
        if time < 60:
            return f"{time:.3f}s"
        else:
            return f"{time/60:.3f}m"

    def on_train_batch_begin(self, batch, logs):
        self.batch_begin_times[batch] = time.time()

    def on_train_batch_end(self, batch, logs):
        now = time.time()
        time_delta = now - self.batch_begin_times[batch]
        total_time = now - self.epoch_begin_time

        # Delete the old batch key.
        del self.batch_begin_times[batch]

        logging.info(
            f"Finished train batch {batch}/{self.training_batches} epch {self.current_epoch} in {self.format_time(time_delta)} / {self.format_time(total_time)} -- " + self.format_batch_logs(logs))

    def on_test_batch_end(self, batch, logs):
        logging.info(
            f"Finished test batch {batch}/{self.validation_batches} epch {self.current_epoch} -- " + self.format_batch_logs(logs))

    def on_epoch_begin(self, epoch, logs):
        self.current_epoch = epoch
        self.epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs):
        now = time.time()
        epoch_time = now - self.epoch_begin_time

        logging.info(
            f"Finished epoch {epoch} in {self.format_time(epoch_time)}, logs: {str(logs)}")

        if logs['val_loss'] <= self.best_val:
            self.best_val = logs['val_loss']
            logging.info("Epoch produced best validation loss so far, saving.")
            self.model.save(f'{self.jobd}/best-val-loss-epch-{epoch}.h5')

        if logs['loss'] < self.best and logs['loss'] < logs['val_loss']:
            self.best = logs['loss']
            logging.info(
                "Epoch produced best training loss so far (with good validation loss), saving.")
            self.model.save(f'{self.jobd}/best-loss-epch-{epoch}.h5')

        gc.collect()


def get_callbacks(job_dir, training_gen, validation_gen):
    # Set up callbacks - one for tensorboard
    tfcallback = keras.callbacks.TensorBoard(
        log_dir=job_dir + "/tensorboard/", profile_batch=0)

    # Another for logging data to CSV
    csvlogger = keras.callbacks.CSVLogger(job_dir + "/training.log")

    # Another for saving checkpoints.
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=job_dir +
        "/model-epoch-{epoch:02d}-loss-{val_loss:.8f}.hdf5",
        save_best_only=False,
        verbose=1,
        moditor='val_loss',
        mode='min',
        save_weights_only=False,
        save_freq='epoch'
    )

    # And another for our custom callback that logs updates
    bcallback = CustomCallback(
        job_dir, training_gen.batch_count(), validation_gen.batch_count())

    # return them to the caller.
    return [tfcallback, csvlogger, model_checkpoint_callback, bcallback]


def create_model(job_dir):
    model = None
    initial_epoch = None

    # Try and find a model in the log directory from which to resume
    logging.info("Finding latest model as starting point")
    # List the models in the job_dir, and find the most recent.
    (model_file, epoch, lss) = tmodel.find_best_model_in_directory(job_dir)
    if model_file is not None:
        # If we have successfully found a model, load it.
        logging.info(
            f"Resuming from model file: {model_file}, epoch {epoch}, loss {lss}")
        model = tmodel.load_model(model_file)
        initial_epoch = epoch
    else:
        # We can't find a model, so generate from scratch.
        logging.info(
            "No existing models found, starting from scratch by generating a model")
        model = tmodel.gen_latest_model()
        initial_epoch = 0

    print(model.summary())

    # Compile the model
    # opt = keras.optimizers.SGD(
    #     lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    # opt = tf.keras.optimizers.Adamax()

    # opt = tf.keras.optimizers.Adagrad()

    # opt = keras.optimizers.Adam(epsilon=1e-8)

    opt = tf.keras.optimizers.Adadelta(learning_rate=0.01)

    logging.info("Compiling model")
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mse', 'mae', 'msle', 'mape'])

    return (model, initial_epoch)


def get_generators(cache_dir="/tmp", ellington_lib="data/example.el"):
    # Set up the data input etc.
    logging.info(f"Loading overall ellington library from {ellington_lib}")

    # Load an overall ellington library
    overall_library = EllingtonLibrary.from_file(ellington_lib)
    # Split it into training and validation
    (train_lib, valid_lib) = overall_library.split_training_validation(ratio=5)

    # Augment the validation training library - and check that the variants are valid.
    # train_lib.augment_library(
    #     config.augmentation_variants, validate_lengths=False)

    # Get the lengths of the sub-libraries
    train_lib_len, valid_lib_len = len(train_lib.tracks), len(valid_lib.tracks)

    logging.info(
        f"Split overall library into (training, validation) sub-libraries of lengths: ({train_lib_len}, {valid_lib_len})")

    logging.debug("Training library: ")
    for trackix in range(train_lib_len):
        track = train_lib.tracks[trackix]
        logging.debug(f"- {trackix}/{train_lib_len}  --  {track.trackname}")

    logging.debug("Validation library: ")
    for trackix in range(valid_lib_len):
        track = valid_lib.tracks[trackix]
        logging.debug(f"- {trackix}/{valid_lib_len}  --  {track.trackname}")

    # Set up the generators to yield training data
    training_gen = LibraryIterator(
        train_lib, start_cutoff=5, end_cutoff=5, multiplier=1)
    validation_gen = LibraryIterator(
        valid_lib, start_cutoff=5, end_cutoff=5, multiplier=1)

    return (training_gen, validation_gen)


def main(cache_dir="/tmp", ellington_lib="data/example.el", job_dir="job"):

    logging.info("Starting training application...")
    config.cache_directory = cache_dir

    # Get data and callbacks.
    (training_gen, validation_gen) = get_generators(cache_dir, ellington_lib)
    callbacks = get_callbacks(job_dir, training_gen, validation_gen)

    # Get the model
    (model, initial_epoch) = create_model(job_dir)

    # Train for just 15 iterations, then fail and hope that a script restarts us.
    # Fit the model using all of the above!
    logging.info("Starting training!")
    model.fit(
        # provide input to x as a generator - don't need to specify y
        x=training_gen,
        # Now many batches of samples per epoch?
        steps_per_epoch=training_gen.batch_count(),
        # Train for a single epochs - which should avoid memory leaks
        epochs=initial_epoch + 3,
        # Start with this epoch (1 by default, may be higher if we're resuming training)
        initial_epoch=initial_epoch,
        # Log a line per epoch
        verbose=2,
        # The callbacks that we wish to run each epoch/batch
        callbacks=callbacks,
        # Our dataset for validating the training of the mode.
        validation_data=validation_gen,
        # Use a slightly larger queue than the default
        max_queue_size=32,
        # And a couple more workers than the default
        workers=4
    )

    sys.exit(-1)


def entrypoint():
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


if __name__ == '__main__':
    entrypoint()
