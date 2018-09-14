import os.path
import logging
import librosa
import librosa.display
import numpy as np

from enum import Flag, unique

from ..util.path import Path

class TrackLoadException(Exception):
    """Exception raised for tracks that have not been loaded.

    Attributes:
        track -- track for which data has not been loaded
        message -- explanation of the error
    """

    def __init__(self, track, message):
        self.track = track
        self.message = message

@unique
class CacheLevel(Flag):
    NONE = 1
    WRITE = 2
    READ = 4
    
    def describe(self):
        # self is the member here
        return self.name, self.value

    def __str__(self):
        return 'Cache level: {0}'.format(self.value)

# Track is sort of a super abstraction of an Ellington library item - we can 
# either construct it from an ellington library entry, or from a file path
# and from a bpm.

# Tracks are also used to manage/contain spectrogram data. This can either be 
# loaded directly from an audio file, or loaded from a cached npz file 
# (which is significantly faster, though uses more hard drive space)
class Track: 

    # Default metadata
    path = None
    bpm = None

    caching = True
    cache_root = None
    cache_path = None

    spect = None
    loaded = False

    @classmethod
    def from_library_entry(cls, elentry, caching=CacheLevel.READ | CacheLevel.WRITE, cache_root="/tmp/bellson/"): 
        return Track(elentry.path, elentry.bpm, caching, cache_root)

    @classmethod
    def from_path(cls, path, bpm=None, caching=CacheLevel.READ | CacheLevel.WRITE, cache_root="/tmp/bellson/"):
        return Track(Path(path), bpm, caching, cache_root)


    def __init__(self, path, bpm=None, caching=CacheLevel.READ | CacheLevel.WRITE, cache_root="/tmp/bellson/"): 
        self.path = path
        self.bpm = bpm
        self.caching = caching 
        self.cache_root = cache_root
        self.cache_path = Path(cache_root + "/" + self.path.digest + ".npz")

    def __str__(self): 
        return str(self.path) + ", " + str(self.cache_path)

    # At times we want to clear out the spectrogram data in order to save space
    # This method gives us a chance to explicitly do so, and it is reccomended
    # that the user calls "gc.collect" afterwards to complete the operation
    def clear(self): 
        self.loaded = False
        del self.spect

    # Load a spectrogram from an audio file
    def load(self, samplerate=44100, lowf=64, highf=320): 
        logging.info("Loading spectrogram info")
        if not CacheLevel.READ in self.caching: 
            self.load_from_audio_file(samplerate, lowf, highf)
        else: 
            if os.path.exists(self.cache_path.canonical): 
                logging.info("Reading from cache")
                self.load_cached_spectrogram()
            else:
                logging.info("Reading from audio file")
                self.load_from_audio_file(samplerate, lowf, highf)
                if CacheLevel.WRITE in self.caching: 
                    logging.info("Caching calculated spectrogram value")
                    self.save_cached_spectrogram()
        logging.info("Loaded spectrogram of shape: " + str(self.spect.shape))
        self.loaded = True

    # Load data directly from an audio file
    def load_from_audio_file(self, samplerate=44100, lowf=64, highf=320):
        if (not self.loaded): 
            logging.info("Loading spectrogram data from audio file")

            # fix the sample rate, otherwise we'll never be able to see if caches are correct...
            (y, sr) = librosa.load(self.path.canonical, sr=samplerate, res_type='kaiser_fast')

            S = librosa.stft(y)
            M = librosa.core.magphase(S)[0]
            # Compute amplitude to db, and cut off the high/low frequencies
            self.spect = librosa.amplitude_to_db(M, ref=np.max)[lowf:highf,:]

            logging.info("Audio of shape: " + str(y.shape))
            logging.info("Sample rate: " + str(sr))
            logging.info("Audio of time: " + str(y.shape[0] / sr))

        else: 
            logging.info("Track data already loaded!")

    # TODO: Maybe integrate tensorflow's file loading/saving here?
    def load_cached_spectrogram(self):
        logging.info("Loading spectrogram data from cache")
        with np.load(self.cache_path.canonical) as npzf: 
            self.spect = npzf['spect'] 
        logging.info("Loaded spectrogram data from cache")

    def save_cached_spectrogram(self): 
        logging.info("Saving spectrogram to cache")
        os.makedirs(self.cache_root, exist_ok=True)
        np.savez_compressed(self.cache_path.canonical, spect=self.spect)
        logging.info("Saved spectrogram to cache")
    
    def plot_spectrogram(self, path=None): 
        import matplotlib.pyplot as plt
        logging.info("Saving spectrogram")

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
        if path is None: 
            path = self.cache_path.canonical + ".png"
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def main(file): 
    # Pass one
    spect = Track.from_path(file, bpm=180, caching=CacheLevel.NONE)
    spect.load()
    logging.info(str(spect))

    # Pass two
    spect = Track.from_path(file, bpm=180, caching=CacheLevel.READ | CacheLevel.WRITE)
    spect.load()
    logging.info(str(spect))

    # Pass two.five
    spect2 = Track.from_path(file, bpm=180, caching=CacheLevel.READ | CacheLevel.WRITE)
    spect2.load()
    logging.info(str(spect2))

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
