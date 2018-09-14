#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph

import tensorflow as tf
from tensorflow import keras

import numpy as np

from bellson.datasources.track import Track, Caching
from bellson.datasources.trackiterator import RangeError

def get_sample(spect, start, length): 
    (h, w) = spect.shape
    end = start + length
    if end >= w: 
        raise RangeError("Requested sample end %d is beyond the audio length %d" % (end, w))
    return spect[:, int(start):int(end)]

def main(model, audiofile):
    # Create the model, print info
    model = keras.models.load_model(model)

    # create a track from the audiofile file path, and load spectrogram data
    # Don't cache, as this just takes up file space
    track = Track.from_path(audiofile, caching=CacheLevel.READ)
    track.load()

    # Get the spectrogram data
    (h, w) = track.spect.shape
    # We want data of shape
    input_w = 1720
    input_h = 256

    sixty = 60 * 86

    samples = [] 
    times = [] 
    for i in range(sixty, w - input_w, 43): 
        try:
            sample = get_sample(track.spect, i, input_w)

            maxv = np.max(np.abs(sample))
            data = np.reshape(sample, (input_h, input_w, 1)) / maxv
            times.append(i / 86)
            samples.append(data)
        except RangeError:
            logging.warn("Random range was invalid - continuing to try again")
    
    print("Predicting batch")
    results = model.predict_on_batch(np.array(samples)).flatten().tolist()

    pairs = zip(times, results)

    print("Results: [{}]".format( "\n ".join('(%.2f, %.2f)' % (t, (400 * r)) for (t, r) in pairs) ))
    print("Mean: %.2f" % (np.mean(results) * 400))
    print("Stddev: %.2f" % (np.std(results) * 400))

    print("Geomean: %.2f" % ((np.array(results).prod()**(1.0/len(results))) * 400))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='The model to use for inference')
    parser.add_argument('--audiofile', required=True, help='The audiofile file to analyse')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
