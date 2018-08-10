import librosa
import librosa.display
import numpy as np
import math
import logging

from .ellington_library import Track

SAMPLE_LENGTH = 10
SAMPLE_START = 60
SAMPLE_INTERVAL = 1

SAMPLE_RATE = 44100

class SamplingConfig: 
    def __init__(self, length, start, interval, rate): 
        self.length = length
        self.start = start 
        self.interval = interval
        self.rate = rate

def is_test(s):
    if s % 10 == 0: 
        return True
    else: 
        return False


class AudioTrack:
    # Meta, and file information
    track = None
    length = None

    # Audio data and sample rate 
    audio = None 
    spect = None
    sr = None 
    loaded = False

    def __init__(self, etrack):
        self.track = etrack
        self.length = librosa.core.get_duration(filename=etrack.filename)    

    # Generate intervals in the audio data, if training, return only if "is_test" is true for s
    def intervals(self, testing=False):
        # Check for tracks shorter than the training sample length
        if self.length < SAMPLE_LENGTH:
            logging.error("Audio track too short to extract intervals")
        # Define the start and end of the range of samples
        start = int(math.floor(min(SAMPLE_START, self.length-SAMPLE_LENGTH)))
        stop = int(math.floor(self.length)) - SAMPLE_LENGTH
        for s in range(start, stop, SAMPLE_INTERVAL):
            # TODO: Handle non-1 SAMPLE_INTERVALs? Do we need to?
            if is_test(s) == testing:
                yield (s, s + SAMPLE_LENGTH)

    def load(self): 
        # Get the audio and sample rate
        # This test is garbage.
        if (not self.loaded): 
            logging.debug("Loading audio data")
            (y, sr) = librosa.load(self.track.filename, sr=SAMPLE_RATE, res_type='kaiser_fast')
            self.audio = y
            self.sr = sr

            # Compute a spectrogram
            S = librosa.stft(self.audio)
            M = librosa.core.magphase(S)[0]
            self.spect = librosa.amplitude_to_db(M, ref=np.max)

            logging.debug("Audio of shape: " + str(self.audio.shape))
            logging.debug("Audio of time: " + str(self.audio.shape[0] / SAMPLE_RATE))
            logging.debug("Spectrogram of shape: " + str(self.spect.shape))

            self.loaded = True
        else: 
            logging.debug("Audio data already loaded!")
        

    def plot_spectrogram(self): 
        import matplotlib.pyplot as plt
        logging.info("Saving spectrograms")

        # Create a figure, and configure it
        (h, w) = self.spect.shape
        fig = plt.figure(figsize=(w/100, h/100))
        ax = plt.subplot(111)
        ax.set_frame_on(False)
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # Perform some magnitue-to-frequency calculations, and write the result to the figure
        librosa.display.specshow(self.spect, y_axis='linear')

        # Save the figure, and close it
        fig.savefig("data/spect/" + self.track.trackname + ".png", dpi=100, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

    def save_spectrogram(self):
        logging.info("Saving spectrogram as numpy array") 
        np.savez_compressed("data/np/" + self.track.trackname + ".npz", spect=self.spect)
        logging.info("Saved to file") 
        logging.info("Loading back from file") 
        np.load("data/np/" + self.track.trackname + ".npz")
        logging.info("Loaded back")

        
    def audio_intervals(self, testing=False):
        self.load()
        logging.debug("Data loaded") 
        for (s, e) in self.intervals(testing): 
            sf = s * self.sr
            ef = e * self.sr
            yield (s, e, self.audio[sf:ef])

        logging.debug("Yielded all audio data available") 

    def spect_intervals(self, testing=False): 
        self.load()
        logging.debug("Data loaded") 
        (h, w) = self.spect.shape
        for (s, e) in self.intervals(testing): 

            ss = int(math.floor((s / self.length) * w))
            # We can't calculate the end in the same way, as we want it to always be the same size: 
            # Todo: Find a better way of calculating this! 
            se = ss + int(math.floor((SAMPLE_LENGTH/ self.length) * w))
            yield (s, e, self.spect[:, ss:se])

        logging.debug("Yielded all spectrogram data available") 

