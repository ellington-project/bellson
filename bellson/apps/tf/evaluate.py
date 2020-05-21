#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import math

from os import path
import sys

import tensorflow as tf
from tensorflow import Graph
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ...libbellson.ellington_library import EllingtonLibrary, Track
from ...libbellson.library_iterator import LibraryIterator, TrackIterator
from ...libbellson import config

import re
import operator


def bpm_mean_error(y_true, y_pred):
    y_true_bpm = tf.multiply(400.0, y_true)
    y_pred_bpm = tf.multiply(400.0, y_pred)

    delta = tf.abs(tf.subtract(y_true_bpm, y_pred_bpm))
    return tf.reduce_mean(delta, axis=-1)


def bpm_std_deviation(y_true, y_pred):
    y_true_bpm = tf.multiply(400.0, y_true)
    y_pred_bpm = tf.multiply(400.0, y_pred)

    delta = tf.abs(tf.subtract(y_true_bpm, y_pred_bpm))
    return tf.math.sqrt(tf.math.reduce_variance(delta))


def evaluate_model(library_generator, modelfile):
    graph1 = tf.Graph()
    with graph1.as_default():
        logging.info("Loading model")
        model = None
        try:
            model = keras.models.load_model(modelfile, compile=False)
        except Exception as e:
            logging.error(f"Threw: {str(e)}")
            logging.info("Manually creating model")
            from trainer.model import model_gen
            input_time_dim = 1720
            input_freq_dim = 256
            model = model_gen(input_time_dim, input_freq_dim)
            model.load_weights(modelfile)

        opt = keras.optimizers.SGD(
            lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

        logging.info("Compiling model")
        model.compile(optimizer=opt,
                      loss='mse',
                      metrics=['mae', 'msle', 'mape', bpm_mean_error, bpm_std_deviation])

        logging.info("Loaded model")

        metrics = model.evaluate(library_generator, verbose=1)
        keys = ["loss", "mae", "msle", "mape",
                "bpm_mean_error", "bpm_std_deviation"]

        return dict(zip(keys, metrics))


def main(cache_dir, ellington_lib,  models):
    config.cache_directory = cache_dir

    # check whether or not the models exist before we start testing them
    for modelf in models:
        logging.info(f"Checking model file {modelf}")
        if not path.exists(modelf):
            logging.error(f"Model file {modelf} does not exist!")
            sys.exit(-1)

    # Set up the data input etc.
    library = EllingtonLibrary.from_file(ellington_lib)

    generator = LibraryIterator(library)

    results = {}
    for modelf in models:
        metrics = evaluate_model(generator, modelf)
        results[path.basename(modelf)] = metrics

    for model, metrics in results.items():
        print(f"Model {model} -- {metrics}")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True,
                        help='Path to cache directory, for pre-compiled histograms')
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    parser.add_argument("--models", required=True, nargs='+',
                        help='Bellson models to evaluate')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
