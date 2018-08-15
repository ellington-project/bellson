import librosa
import librosa.display
import numpy as np
import math
import logging
import os

from .ellington_library import Track

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
    length = None
    data = None
    loaded = False

    def __init__(self, etrack):
        self.track = etrack
        self.length = librosa.core.get_duration(filename=etrack.filename)    

    def load(self, folder="data/np/"): 
        # Load the spectrogram from a pre-written file
        # This test is garbage.
        if (not self.loaded): 
            filename = folder + self.track.digest + ".npz"
            logging.info("Loading spectrogram from file " + filename) 
            if os.path.exists(filename): 
                with np.load(folder + self.track.digest + ".npz") as npzf:
                    self.data = npzf['spect']
                    logging.info("Loaded spectrogram data")
            else: 
                logging.error("Could not load spectrogram file - have they been generated?")
                raise Exception("Could not load spectrogram data - has it been generated?")
        else: 
            logging.debug("Audio data already loaded.")

    def interval(self, start=60, sample_length=10): 
        # Compute the length of the interval (in seconds) 
        # sample_length = end-start
        end = start + sample_length
        logging.info("Extracting audio interval (" + str(start) + "," + str(end) +") from " + str(sample_length) + " as a spectrogram")
        # Check for tracks shorter than the training sample length
        if self.length < sample_length:
            logging.error("Requested sample length (" + str(sample_length) + ") is longer than the audio length (" + str(self.length) + ")")
            raise RangeError("Requested sample length (" + str(sample_length) + ") is longer than the audio length (" + str(self.length) + ")")
        # Check for samples that go beyond the end of the track        
        if end >= self.length or start >= self.length: 
            logging.error("Requested interval (" + str(end) +"," + str(start) +") goes beyond the end of the track (" + str(self.length) + ")")
            raise RangeError("Requested interval (" + str(end) +"," + str(start) +") goes beyond the end of the track (" + str(self.length) + ")")
        # Check for samples that go beyond the start of the track
        if end < 0 or start < 0: 
            logging.error("Requested interval (" + str(end) +"," + str(start) +") goes beyond the start of the track")
            raise RangeError("Requested interval (" + str(end) +"," + str(start) +") goes beyond the start of the track")
        # Check for samples that 
        # Get the size of the spectrogram data
        (h, w) = self.data.shape
        # Compute the start in terms of the array
        # There should definitely be a better way of doing this.
        start_ix = int(math.floor((start / self.length) * w))
        # Compute the end in terms of the length
        # we want it to be consistent across audio file lengths
        end_ix = start_ix + 861 # int(math.floor((sample_length / self.length) * w))
        logging.info("Extracting data in spectrogram interval (" + str(start_ix) + "," + str(end_ix) +") from " + str(w))
        ret= self.data[:, start_ix:end_ix]
        logging.info("Returned data shape: " + str(ret.shape))
        return ret

    
    def plot_spectrogram(self): 
        import matplotlib.pyplot as plt
        logging.info("Saving spectrograms")

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
