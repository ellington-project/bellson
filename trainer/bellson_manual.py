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
            print('step: {}/{} ::{}'.format(self.step,
                                            self.params['steps'],
                                            metrics_log))
            self.metric_cache.clear()
            gc.collect()
            # objgraph.show_most_common_types(limit=20)
            # objgraph.get_new_ids()


def main(data_dir="data/smnp/", ellington_lib="data/example.el", job_dir="logs"):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)

    logging.info(device_lib.list_local_devices())        

    # Set up the data input etc.
    train_lib = EllingtonLibrary.from_file(ellington_lib)
    valid_lib = EllingtonLibrary.from_file(ellington_lib)

    print("Training with library of size: " + str(len(train_lib.tracks)))
    print("Validating with library of size: " + str(len(valid_lib.tracks)))

    training_gen = LibraryIterator(
        train_lib, data_dir, samples=32, batchsize=32, start=30, end=150, iterations=1)
    validation_gen = LibraryIterator(
        valid_lib, data_dir, samples=8, batchsize=64, start=30, end=150, iterations=1)

    input_time_dim = 1720
    input_freq_dim = 256

    # Create the model, print info
    model = model_gen(input_time_dim, input_freq_dim)
    print(model.summary())

    # Compile the model
    adam = tf.train.AdamOptimizer(learning_rate=1e-04)
    sgd = keras.optimizers.SGD(
        lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=['mae', 'msle', 'mape'])

    tfcallback = keras.callbacks.TensorBoard(log_dir='./logs',
                                           histogram_freq=0,
                                           write_grads=True,
                                           write_graph=False,
                                           write_images=False,
                                           batch_size=16)
    bcallback = NBatchLogger(1)
    model_file_name = 'models/bpm-model' + '-{epoch:03d}-{val_loss:.5f}.h5'
    cplogger = keras.callbacks.ModelCheckpoint(model_file_name, monitor='val_mean_absolute_error', verbose=1, save_best_only=False)

    # Calculate how many epochs we want
    epochs = 100 * len(train_lib.tracks)

    # Star training!
    for epoch in range(0, epochs):
        batch = 0
        # Save the model locally
        model.save('model.h5')

        # Save the model to the Cloud Storage bucket's jobs directory
        print("Saving to : " + job_dir)
        with file_io.FileIO('model.h5', mode='rb') as input_f:
            with file_io.FileIO(job_dir + '-model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())
                
        logging.info("Epoch: %d / %d" % (epoch, epochs))
        for (train, target) in training_gen.batch():
            batch = batch + 1
            metrics = model.train_on_batch(x=train, y=target)

            metrics_log = ''
            for (val, k) in zip(metrics, model.metrics_names):
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            logging.info('step: {}/{} ::{}'.format(batch,
                                            training_gen.len(),
                                            metrics_log))
            gc.collect()

        for (train, target) in validation_gen.batch():
            logging.info("Predicting batch")
            results = model.predict_on_batch(train).flatten().tolist()

            d = {}
            for (e, r) in zip(target, results):
                d.setdefault(e * 400, []).append(r * 400)

            for k, v in d.items():
                mean = np.mean(v)
                vss = ', '.join('%.2f' % i for i in v[0:10])
                logging.info("{:0.4f} : {:0.4f} :: [{}] ".format(k, mean, vss))

            target_mean = np.mean(target) * 400
            target_std = np.std(target) * 400

            result_mean = np.mean(results) * 400
            result_std = np.std(results) * 400

            logging.info("Target: %0.4f, %0.4f" % (target_mean, target_std))
            logging.info("Result: %0.4f, %0.4f" % (result_mean, result_std))

            gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to training data, in the form of compressed numpy arrays')
    parser.add_argument('--ellington-lib', required=True, help='The ellington library from which to read track names and BPMs')
    parser.add_argument('--job-dir', required=True, help='The directory to export the model, and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
