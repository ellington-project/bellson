import librosa
import librosa.display
import numpy as np
import math
import logging

from .ellington_library import Track

SAMPLE_LENGTH = 10
SAMPLE_START = 60
SAMPLE_INTERVAL = 1

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
        
    def audio_intervals(self, testing=False):
        # Get the audio and sample rate
        # This test is garbage.
        if (not self.loaded): 
            logging.debug("Loading audio data")
            (y, sr) = librosa.load(self.track.filename, sr=1000, res_type='kaiser_fast')
            self.audio = y
            self.sr = sr
            self.loaded = True
        else: 
            logging.debug("Audio data already loaded!")

        logging.debug("Data loaded") 
        for (s, e) in self.intervals(testing): 
            sf = s * self.sr
            ef = e * self.sr
            yield (s, e, self.audio[sf:ef])

        logging.debug("Yielded all audio data available") 
