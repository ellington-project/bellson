from .track import Track, TrackLoadException

class RangeError(Exception): 
    """Exception raised for ranged outside the audio range"""
    def __init__(self,  message): 
        self.message=message

class TrackIterator: 
    track = None

    def __init__(self, track): 
        # Set the values, and load the spectrogram
        self.track = track
        if not track.loaded: 
            raise TrackLoadException(track, "Track was not loaded before iterator construction!")

    def __str__(self): 
        return str(self.track)
    # def iter(self):
    #     # Iterate over the range, and emit samples:
    #     i = 0 
    #     # Iterate with a while loop, as any iteration might fail
    #     while i < self.samples: 
    #         s = uniform(self.start, self.end)
    #         try:
    #             print("Yielding data in range (" + str(s) + "," + str(s+ self.length)+")")
    #             data = self.spect.interval(s)
    #             if data.shape == (256, 1720):
    #                 i = i + 1
    #                 yield (self.track.bpm, data)
    #         except RangeError: 
    #              print("Random range was invalid - continuing to try again")
    #     print("Yielded " + str(self.samples) + " samples.")



def main(file): 
    # Pass one
    tr = Track.from_path(file, bpm=180)
    tr.load()
    logging.info(str(tr))

    iter = TrackIterator(tr)
    logging.info(str(iter))

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
