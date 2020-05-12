#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import time

import tensorflow as tf
from tensorflow import keras

import numpy as np

from bellson.audio import Track, CacheLevel, TrackIterator, RangeError


def main(model, audiofile, inc=1):
    start_time = time.time()
    logging.info("Started infererence tool")
    # Create the model, print info
    logging.info("Loading model")
    model = keras.models.load_model(model, compile=False)
    logging.info("Loaded model")
    # create a track from the audiofile file path, and load spectrogram data
    # Don't cache, as this just takes up file space
    track = Track.from_path(audiofile, caching=CacheLevel.READ)
    track.load()
    logging.info(f"Loaded track from path {audiofile}")

    # Create an iterator over the track
    titer = TrackIterator(track)
    (h, w) = titer.config.sample_shape()

    samples = []
    starttimes = []
    for (sample, start, end, bpm) in titer.linear_iter(float(inc)):
        # Perform some noramlisation
        logging.info("Yielded sample")
        maxv = np.max(np.abs(sample))
        data = np.reshape(sample, (h, w, 1)) / maxv
        starttimes.append(start)
        samples.append(data)

    logging.info("Predicting batch")
    results = model.predict_on_batch(
        np.array(samples)).flatten().tolist()

    pairs = zip(starttimes, results)

    print("Results: [{}]".format("\n ".join('(%.2f, %.2f)' %
                                            (t, (400 * r)) for (t, r) in pairs)))
    logging.info("Mean: %.2f" % (np.mean(results) * 400))
    logging.info("Stddev: %.2f" % (np.std(results) * 400))

    logging.info("Geomean: %.2f" %
                 ((np.array(results).prod()**(1.0/len(results))) * 400))

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='The model directory to use for inference')
    parser.add_argument('--audiofile', required=True,
                        help='The audiofile file to analyse')
    parser.add_argument('--inc', required=True,
                        help='The inc for the audio iterator')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
