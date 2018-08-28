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
from trainer.audio import Audio, BareTrack
from trainer.generator import LibraryIterator, TrackIterator
from trainer.spectrogram import RangeError
from trainer.model import model_gen

def get_sample(spect, start, length): 
    (h, w) = spect.shape
    end = start + length
    if end >= w: 
        raise RangeError("Requested sample end %d is beyond the audio length %d" % (end, w))
    return spect[:, int(start):int(end)]




def main(model, audio):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)

    # Create the model, print info
    model = keras.models.load_model(model)
    print(model.summary())

    # Create the bare track to load spectrogram data from
    track = BareTrack(audio)
    # And create an Audio object from it
    audio = Audio(track)
    audio.load() # Load the data, and compute a spectrogram. 
    audio.plot_spectrogram()

    # Get the spectrogram data, and cut off frequencies
    spect = audio.spect[64:320,:]
    (h, w) = spect.shape
    # We want data of shape
    input_w = 1720
    input_h = 256

    sixty = 60 * 86

    
    samples = [] 
    times = [] 
    for i in range(sixty, w - input_w, 43): 
        try:
            sample = get_sample(spect, i, input_w)

            maxv = np.max(np.abs(sample))
            data = np.reshape(sample, (input_h, input_w, 1)) / maxv
            times.append(i / 86)
            samples.append(data)
        except RangeError:
            print("Random range was invalid - continuing to try again")
    
    print("Predicting batch")
    results = model.predict_on_batch(np.array(samples)).flatten().tolist()

    pairs = zip(times, results)

    print("Results: [{}]".format( "\n ".join('(%.2f, %.2f)' % (t, (400 * r)) for (t, r) in pairs) ))
    print("Mean: %.2f" % (np.mean(results) * 400))
    print("Stddev: %.2f" % (np.std(results) * 400))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='The model to use for inference')
    parser.add_argument('--audio', required=True, help='The audio file to analyse')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
