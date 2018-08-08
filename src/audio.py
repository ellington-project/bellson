import librosa
import librosa.display
import numpy as np
import math

from ellington_library import Track

SAMPLE_LENGTH = 60
SAMPLE_START = 60
SAMPLE_INTERVAL = 1

def is_training(s):
    if s % 10:
        return True
    else:
        return False

def is_test(s):
    return not is_training(s)

class AudioTrack:
    track = None
    length = None

    def __init__(self, etrack):
        self.track = etrack
        self.length = librosa.core.get_duration(filename=etrack.filename)    

    # Generate intervals in the audio data
    def intervals(self):
        # Check for tracks shorter than the training sample length
        if self.length < SAMPLE_LENGTH:
            print("Audio track too short to extract intervals")
        # Define the start and end of the range of samples
        start = int(math.floor(min(SAMPLE_START, self.length-SAMPLE_LENGTH)))
        stop = int(math.floor(self.length)) - SAMPLE_LENGTH
        for s in range(start, stop, SAMPLE_INTERVAL):
            # TODO: Handle non-1 SAMPLE_INTERVALs? Do we need to?
            yield (s, s+ SAMPLE_LENGTH)

    def train_intervals(self):
        for (s, e) in self.intervals():
            if is_training(s): 
                yield (s, e)

    def test_intervals(self):
        for (s, e) in self.intervals(): 
            if is_test(s): 
                yield (s, e)