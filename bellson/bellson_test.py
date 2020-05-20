#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import time

import tensorflow as tf
from tensorflow import keras

import numpy as np

from common.ellington_library import EllingtonLibrary, Track
from trainer.library_iterator import LibraryIterator, TrackIterator
import trainer.model as tmodel
import common.config as config


def main(libraryfile):
    config.cache_directory = "/mnt/bigboi/training_cache/"

    overall_library = EllingtonLibrary.from_file(libraryfile)

    validation_gen = LibraryIterator(overall_library)

    for i in range(0, len(validation_gen)):
        batch = validation_gen[i]


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--libraryfile', required=True,
                        help='The model directory to use for inference')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
