import librosa
# import librosa.display
import numpy as np
import math
import logging
import os
from io import BytesIO
import pickle
from tensorflow.python.lib.io import file_io
from tensorflow import __version__ as tf_version

from trainer.ellington_library import Track

SAMPLE_LENGTH = 10
SAMPLE_START = 60
SAMPLE_INTERVAL = 1

SAMPLE_RATE = 44100

class RangeError(Exception): 
    """Exception raised for ranged outside the audio range"""
    def __init__(self,  message): 
        self.message=message

class Spectrogram:
    # Meta, and file information
    track = None
    # length = None
    samples = None
    data = None
    loaded = False    

    def __init__(self, etrack):
        self.track = etrack
        # self.length = librosa.core.get_duration(filename=etrack.filename)    

    def load(self, folder="data/np/"): 
        # Load the spectrogram from a pre-written file
        # This test is garbage.
        if (not self.loaded): 
            filename = folder + "/" + self.track.digest + ".npz"
            print("Loading spectrogram from file " + filename) 
            if tf_version >= '1.1.0':
                mode = 'rb'
            else: # for TF version 1.0
                mode = 'r'
            f = file_io.FileIO(filename, mode)
            
            with np.load(BytesIO(f.read())) as npzf:
                self.data = npzf['spect']
                print("Loaded spectrogram data")

            self.samples = self.data.shape[1]

        else: 
            print("Audio data already loaded.")

    def interval(self, start=60, samples=1720): 
        # Compute the length of the interval (in seconds) 
        # sample_length = end-start
        # end = start + sample_length
        # print("Extracting audio interval (" + str(start) + "," + str(end) +") from " + str(sample_length) + " as a spectrogram")
        start_ix = int(start * 86) # hardcode this for now
        end_ix = start_ix + samples
        # Check for tracks shorter than the training sample length
        if self.samples < samples:
            print("Requested sample length (" + str(self.samples) + ") is longer than the audio length (" + str(samples) + ")")
            raise RangeError("Requested sample length (" + str(self.samples) + ") is longer than the audio length (" + str(samples) + ")")

        # Check for samples that go beyond the end of the track        
        if end_ix >= self.samples or start_ix >= self.samples: 
            print("Requested interval (" + str(start_ix) +"," + str(end_ix) +") goes beyond the end of the track (" + str(self.samples) + ")")
            raise RangeError("Requested interval (" + str(start_ix) +"," + str(end_ix) +") goes beyond the end of the track (" + str(self.samples) + ")")
        # Check for samples that go beyond the start of the track
        if end_ix < 0 or start_ix < 0: 
            print("Requested interval (" + str(end_ix) +"," + str(start_ix) +") goes beyond the start of the track")
            raise RangeError("Requested interval (" + str(end_ix) +"," + str(start_ix) +") goes beyond the start of the track")
        # Check for samples that 
        # Get the size of the spectrogram data
        # (h, w) = self.data.shape
        # Compute the start in terms of the array
        # There should definitely be a better way of doing this.
        # start_ix = int(math.floor((start / self.length) * w))
        # Compute the end in terms of the length
        # we want it to be consistent across audio file lengths
        # end_ix = start_ix + 1720 # int(math.floor((sample_length / self.length) * w))
        print("Extracting data in spectrogram interval (" + str(start_ix) + "," + str(end_ix) +") from " + str(self.samples))
        ret= self.data[:, start_ix:end_ix]
        print("Returned data shape: " + str(ret.shape))
        return ret

    
    def plot_spectrogram(self): 
        import matplotlib.pyplot as plt
        print("Saving spectrograms")

        # Create a figure, and configure it
        (h, w) = self.data.shape
        fig = plt.figure(figsize=(w/100, h/100))
        ax = plt.subplot(111)
        ax.set_frame_on(False)
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # Perform some magnitue-to-frequency calculations, and write the result to the figure
        librosa.display.specshow(self.data, y_axis='linear')

        # Save the figure, and close it
        fig.savefig("data/spect/" + self.track.trackname + ".png", dpi=100, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
