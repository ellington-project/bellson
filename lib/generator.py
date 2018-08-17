import logging
import copy
import random 
# from numpy import array
import numpy as np
from numpy.random import uniform
import gc

from .ellington_library import EllingtonLibrary, Track
from .spectrogram import Spectrogram, RangeError

class TrackIterator: 
    track = None
    spect = None
    def __init__(self, track, start=60, end=180, length=10, samples=10): 
        # Set the values, and load the spectrogram
        self.track = track
        self.spect = Spectrogram(track)
        self.spect.load(folder="data/smnp/")

        # Set the config values
        self.start = start 
        self.end = end
        self.length = length
        self.samples = samples

    def iter(self):
        # Iterate over the range, and emit samples:
        i = 0 
        # Iterate with a while loop, as any iteration might fail
        while i < self.samples: 
            s = uniform(self.start, self.end)
            try:
                logging.info("Yielding data in range (" + str(s) + "," + str(s+ self.length)+")")
                data = self.spect.interval(s, self.length)
                if data.shape == (256, 860):
                    i = i + 1
                    yield (self.track.bpm, data)
            except RangeError: 
                 logging.warn("Random range was invalid - continuing to try again")
        logging.info("Yielded " + str(self.samples) + " samples.")


class LibraryIterator: 
    library = None
    
    # Initialise a generator from a library
    def __init__(self, library, start=60, end=180, length=10, samples=10, iterations=10, batchsize=10): 
        # Make a deep copy of the library so that we can shuffle it. 
        self.library = copy.deepcopy(library)
        # Cache the config values
        self.start = start 
        self.end = end 
        self.length = length 
        self.samples = samples
        self.iterations = iterations
        self.batchsize = batchsize
    
    def len(self): 
        return len(self.library.tracks) * self.samples

    def shuffle(self): 
        logging.info("Shuffling library")
        random.shuffle(self.library.tracks)
    
    def iter(self): 
        # Go across <self.iterations> iterations
        while True:
            # Start by shuffling the library
            self.shuffle()
            # Iterate over the tracks, and get 20 random samples. 
            for t in self.library.tracks:
                logging.info("Yielding spectrogram data for " + t.trackname)
                ti = TrackIterator(t, self.start, self.end, self.length, self.samples)
                # Generate <samples> random samples from the track, and yield them
                for s in ti.iter():                     
                    yield s 
                del ti

    def batch(self): 
        # Yield an iterator over batches of samples 
        # Iterate over the iterator: 
        ix = 0
        inputs  = []
        targets = []  
        for s in self.iter(): 
            target = float(s[0]) / 400.0
            targets.append(target)

            inp = s[1]
            (w, h) = inp.shape
            data = np.reshape(inp, (w, h, 1))
            inputs.append(data)

            ix = ix + 1

            if( ix == self.batchsize):                 
                inputs_arr = np.stack(inputs, axis=0)
                targets_arr = np.stack(targets, axis=0)
                
                logging.info("Yielding an array of " + str(inputs_arr.shape) + " samples")

                yield inputs_arr, targets_arr

                del inputs 
                del targets
                gc.collect()
                inputs = []
                targets = [] 
                ix = 0
        

