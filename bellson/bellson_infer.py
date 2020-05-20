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
from trainer.library_iterator import TrackIterator


def main(modelfile, audiofile):
    start_time = time.time()
    logging.info("Started infererence tool")
    # Create the model, print info
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

    logging.info("Loaded model")

    # create a track from the audiofile file path, and load spectrogram data
    # Don't cache, as this just takes up file space
    track_iterator = TrackIterator.from_filename(audiofile)
    logging.info(f"Loaded track from path {audiofile}")

    (starttimes, samples) = track_iterator.get_batch_with_start_times(sample_count=100)

    logging.info("Predicting batch")
    results = model.predict_on_batch(samples).flatten().tolist()

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
    parser.add_argument('--modelfile', required=True,
                        help='The model directory to use for inference')
    parser.add_argument('--audiofile', required=True,
                        help='The audiofile file to analyse')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
