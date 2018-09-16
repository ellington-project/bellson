import logging

import numpy
from numpy.random import uniform

from .track import Track, TrackLoadException
from .configuration import Configuration

class RangeError(Exception): 
    """Exception raised for ranged outside the audio range"""
    def __init__(self,  message): 
        self.message=message

class TrackIterator: 
    track = None
    start = None
    end = None
    samples = None 
    config = None


    def __init__(self, track, start=30, end=180, samples=10, config=Configuration.default()): 
        # Set the values, and load the spectrogram
        self.track = track
        self.start = start 
        self.end = end 
        self.samples = samples
        self.config = config
        if not track.loaded: 
            raise TrackLoadException(track, "Track was not loaded before iterator construction!")

    def __str__(self): 
        return str(self.track)


    def interval(self, start): 
        # Start by getting the size of our spectrogram, and our final index
        (h, w) = self.track.spect.shape
        end = start + self.config.sample_frames()
        logging.info("Extracting interval: %d, %d" % (start, end))
        # Check that our start is not less than zero
        if start < 0 or end < 0: 
            raise RangeError("Requested interval (%d, %d) goes beyond the start of the track" % (start, end))

        # check that it does not go beyond the end of the track 
        if start >= w or end >= w: 
            raise RangeError("Requested interval (%d, %d) goes beyond the end of the track" % (start, end))

        ret= self.track.spect[:, start:end]
        logging.info("Extracted interval of shape: " + str(ret.shape))
        return ret


    # TODO: How can we abstract these two? 
    def linear_iter(self, increment=1): 
        # Iterate over the range, and emit samples:
        i = 0
        # We don't care about iterations failing, so stick it in a for
        # for s in range(self.start, self.end-self.config.samplelen, increment): 
        for s in numpy.arange(self.start, self.end-self.config.samplelen, increment): 
            e = s + self.config.samplelen
            try: 
                logging.info("Trying range: %.2f, %.2f" % (s, e))
                fstart = self.config.seconds_to_frames(s)
                data = self.interval(fstart)
                if data.shape == self.config.sample_shape():
                    logging.info("Shape consistent and correct!")
                    i = i + 1
                    yield (data, s, e, self.track.bpm)
                else: 
                    logging.error("Data shape {} does not match expected shape {} ".format(data.shape, self.config.sample_shape()))
            except RangeError: 
                 print("Random range was invalid - continuing to try again")
        logging.info("Yielded %d samples." % i)


    def random_iter(self):
        # Iterate over the range, and emit samples:
        i = 0 
        # Iterate with a while loop, as any iteration might fail
        while i < self.samples: 
            s = uniform(self.start, self.end)
            e = s + self.config.samplelen
            try:
                logging.info("Trying range: %.2f, %.2f" % (s, e))
                fstart = self.config.seconds_to_frames(s)
                data = self.interval(fstart)
                if data.shape == self.config.sample_shape():
                    logging.info("Shape consistent and correct!")
                    i = i + 1
                    yield (data, s, e, self.track.bpm)
                else: 
                    logging.error("Data shape {} does not match expected shape {} ".format(data.shape, self.config.sample_shape()))
            except RangeError: 
                 print("Random range was invalid - continuing to try again")
        print("Yielded %d samples." % i)

def main(file): 
    # Pass one
    tr = Track.from_path(file, bpm=180)
    tr.load()
    logging.info(str(tr))

    triter = TrackIterator(tr)
    logging.info(str(triter))

    logging.info(str(triter.interval(60*84).shape))

    for (sample, start, end, bpm) in triter.random_iter(): 
        logging.info("Bpm: %d" % (bpm))
        logging.info("Shape: %d, %d" % sample.shape)

    for (sample, start, end, bpm) in triter.linear_iter(1): 
        logging.info("Bpm: %d" % (bpm))
        logging.info("Shape: %d, %d" % sample.shape)

if __name__ == '__main__':
    import logging
    import argparse
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to load as a spectrogram file')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
