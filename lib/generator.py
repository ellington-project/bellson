import logging
import copy
import random 
from numpy import array
from numpy.random import uniform

from .ellington_library import EllingtonLibrary, Track
from .spectrogram import Spectrogram, RangeError

class TrackIterator: 
    track = None
    spect = None
    def __init__(self, track, start=60, end=180, length=10, samples=10): 
        # Set the values, and load the spectrogram
        self.track = track
        self.spect = Spectrogram(track)
        self.spect.load()

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
                data = self.spect.interval(s, s + self.length)
                i = i + 1
                yield (self.track.bpm, data)
            except RangeError: 
                 logging.info("Random range was invalid - continuing to try again")
        logging.info("Yielded " + str(self.samples) + " samples.")


class LibraryIterator: 
    library = None    

    # Initialise a generator from a library
    def __init__(self, library, start=60, end=180, length=10, samples=10, iterations=10, batchsize=100): 
        # Make a deep copy of the library so that we can shuffle it. 
        self.library = copy.deepcopy(library)
        # Cache the config values
        self.start = start 
        self.end = end 
        self.length = length 
        self.samples = samples
        self.iterations = iterations
        self.batchsize = 10
    
    def len(self): 
        return len(self.library.tracks) * self.samples * self.iterations * self.batchsize

    def shuffle(self): 
        logging.info("Shuffling library")
        random.shuffle(self.library.tracks)
    
    def iter(self): 
        # Go across <self.iterations> iterations
        for i in range(0, self.iterations):
            # Start by shuffling the library
            self.shuffle()
            # Iterate over the tracks, and get 20 random samples. 
            for t in self.library.tracks:
                logging.info("Yielding spectrogram data for " + t.trackname)
                ti = TrackIterator(t, self.start, self.end, self.length, self.samples)
                # Generate <samples> random samples from the track, and yield them
                for s in ti.iter(): 
                    yield s 

    def batch(self): 
        # Yield a batch of samples 
        inputs  = []
        targets = []        
        for i in range(self.batchsize):
            p = next(self.iter())
            targets.append(p[0])
            inputs.append(p[1])
        
        print("Inputs length: " + len(inputs.len))
        return (array(inputs), array(targets))



