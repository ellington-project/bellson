#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import time

import tensorflow as tf
from tensorflow import keras

import numpy as np

# from bellson.audio import Track, CacheLevel, TrackIterator, RangeError
from ...libbellson.library_iterator import TrackIterator
from ...libbellson.model import load_model


def main(modelfile, audiofile):
    start_time = time.time()
    logging.info("Started infererence tool")
    # Create the model, print info
    logging.info("Loading model")
    model = load_model(modelfile)

    logging.info("Loaded model")

    # create a track from the audiofile file path, and load spectrogram data
    # Don't cache, as this just takes up file space
    track_iterator = TrackIterator.from_filename(audiofile)
    logging.info(f"Loaded track from path {audiofile}")

    samples = track_iterator.get_uniform_batch(sample_c=100)

    logging.info("Predicting batch")
    results = model.predict_on_batch(samples).flatten().tolist()

    logging.debug("Results: [{}]".format("\n ".join(
        ['%.2f' % (r * 400) for r in results])))
    logging.info("Mean: %.2f" % (np.mean(results) * 400))
    logging.info("Stddev: %.2f" % (np.std(results) * 400))

    logging.info("Geomean: %.2f" %
                 ((np.array(results).prod()**(1.0/len(results))) * 400))

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', required=True,
                        help='The model directory to use for inference')
    parser.add_argument('--audiofile', required=True,
                        help='The audiofile file to analyse')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
