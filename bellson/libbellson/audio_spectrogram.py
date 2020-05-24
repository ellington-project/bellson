import logging
import librosa
import librosa.display
import numpy as np
import math
import logging
import hashlib
import os.path

from .ellington_library import Track
from . import config


def time_to_stft_frame(time):
    # First calculate time to sample:
    # sample_ix = time * config.SAMPLE_RATE
    # Then use that to calculate the column, by dividing by the hop_length
    # stft_col = sample_ix / config.HOP_LENGTH
    # return int((time * config.SAMPLE_RATE)/config.HOP_LENGTH)

    # To calculate the starting (frame) index for a time t, call:
    return librosa.core.time_to_frames(np.array([time]), sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH)[0]


def stft_frame_to_time(col_ix):
    # First, calculate column IX to sample_ix
    # sample_ix = col_ix * config.HOP_LENGTH
    # Then divide by the sample rate to get time
    # time = sample_ix / config.SAMPLE_RATE
    # return float(col_ix * config.HOP_LENGTH) / float(config.SAMPLE_RATE)

    # To calculate the time of a (frame) index i, call:
    return librosa.core.frames_to_time(np.array([col_ix]), sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH)[0]


class RangeError(Exception):
    """Exception raised for ranged outside the audio range"""

    def __init__(self,  message):
        self.message = message


class AudioSpectrogram:
    # The filename of the audio file which we used to read audio.
    source_filename = None
    # The filename of the cache of the spectrogram that we use to generate it.
    cache_name = None

    # raw spectrogram data
    spect_data = None

    # Tempo modifier - how much we should alter the tempo after creating the spectrogram
    tempo_modifier = None

    def __init__(self, audio_file, track_digest, should_load=True, read_from_cache=True, tempo_modifier=None):
        logging.debug("Creating AudioSpectrogram(" +
                      audio_file + ", " + track_digest + ")")
        logging.debug("Cache directory: " + config.cache_directory)
        self.source_filename = os.path.abspath(audio_file)
        self.cache_name = os.path.abspath(
            config.cache_directory + "/" + track_digest + ".npz")
        self.tempo_modifier = tempo_modifier
        if should_load:
            self.load_spectrogram()

    @classmethod
    def from_library_track(cls, track, should_load=True, read_from_cache=True, tempo_modifier=None):
        return AudioSpectrogram(track.filename, track.digest, should_load, read_from_cache)

    @classmethod
    def from_file_name(cls, filename, should_load=True, read_from_cache=True, tempo_modifier=None):
        return AudioSpectrogram(filename, hashlib.sha256(filename.encode('utf-8')).hexdigest(), should_load, read_from_cache)

    def clear(self):
        del self.spect_data
        self.spect_data = None

    def generate_spectrogram(self, should_cache=True):
        """
            Generate and store a spectrogram from the audio file

            Loads audio data from self.source_filename, and stores a computed spectrogram in self.spect_data. 

            Does no caching/saving/plotting - this is a "raw" interface to generating a spectrogram. 
        """
        logging.debug("Loading audio data")
        (y, sr) = librosa.load(self.source_filename,
                               sr=config.SAMPLE_RATE, res_type='kaiser_fast')

        assert sr == config.SAMPLE_RATE

        # Compute a spectrogram,
        S = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                         win_length=config.WIN_LENGTH)
        M = librosa.core.magphase(S)[0]

        # store it as self.spect_data, after cutting off unneccessary high/low frequencies
        self.spect_data = librosa.amplitude_to_db(M, ref=np.max)

        # if we should modify the tempo, do so to the computed spectrogram
        # Use the high quality pythonrubberband method if possible
        if self.tempo_modifier is not None:
            try:
                import pyrubberband
                self.spect_data = pyrubberband.pyrb.time_stretch(
                    self.spect_data, sr, self.tempo_modifier)
            except:
                # Fall back to low quality librosa
                self.spect_data = librosa.effects.time_stretch(
                    self.spect_data, self.tempo_modifier)

        logging.debug(
            f"spectrogram shape before low/high cutoff: {self.spect_data.shape}")
        self.spect_data = self.spect_data[
            config.SPECTROGRAM_LOW_FREQUENCY_CUTOFF:config.SPECTROGRAM_HIGH_FREQUENCY_CUTOFF, :]
        logging.debug(
            f"spectrogram shape after low/high cutoff: {self.spect_data.shape}")

        # Normalise the values
        self.spect_data = self.spect_data / np.max(np.abs(self.spect_data))

        logging.debug("Audio of shape: " + str(y.shape))
        logging.debug("Audio of time: " + str(y.shape[0] / config.SAMPLE_RATE))
        logging.debug("Spectrogram of shape: " + str(self.spect_data.shape))
        logging.debug("Calculated time (from spectrogram): " +
                      str(self.audio_length()))
        logging.debug("Calculated frames (from time): " +
                      str(time_to_stft_frame(y.shape[0] / config.SAMPLE_RATE)))

        # If we should cache, save it.
        if should_cache:
            logging.debug("Saving spectrogram to cache. ")
            np.savez_compressed(self.cache_name, spect=self.spect_data)

    def load_spectrogram(self, read_from_cache=True):
        """
            Load a spectrogram for an associated audio file.  

            If we have spectrogram data already, this is basically a noop, 
            but otherwise we try and load from cache, and then finally try and compute the data. 
        """
        # if we already have data, we're done.
        if self.spect_data is not None:
            logging.debug("Spectrogram already loaded, finishing.")
            return

        # if we've been told not to use the cache, do that.
        if not read_from_cache:
            logging.debug("Manually generating spectrogram")
            self.generate_spectrogram()
            return

        # else, check to see if we can load it from the cache.
        try:
            with np.load(self.cache_name) as npzf:
                self.spect_data = npzf['spect']
            logging.debug("Spectrogram loaded from cache " + self.cache_name)
        except:
            # If that fails, fall back to manually generating it.
            logging.debug("Manually generating spectrogram")
            self.generate_spectrogram()

    def plot_spectrogram(self, filename=None):
        import matplotlib.pyplot as plt

        if filename is None:
            filename = self.cache_name + ".png"

        logging.debug("Saving spectrogram to " + filename)

        # Create a figure, and configure it
        (h, w) = self.spect_data.shape
        logging.debug(f"Spectrogram shape: ({h}, {w})")
        fig = plt.figure(figsize=(w/100, h/100))
        ax = plt.subplot(111)
        ax.set_frame_on(False)
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # Perform some magnitue-to-frequency calculations, and write the result to the figure
        librosa.display.specshow(self.spect_data, y_axis='linear')

        # Save the figure, and close it
        fig.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

    def audio_length(self):
        return stft_frame_to_time(self.spect_data.shape[1])

    def interval(self, start=60, samples=config.input_time_dim):

        # Compute the length of the interval (in seconds)
        # sample_length = end-start
        # end = start + sample_length
        # logging.info("Extracting audio interval (" + str(start) + "," +
        #              str(end) + ") from " + str(sample_length) + " as a spectrogram")

        # start_ix = int(start * 86)  # hardcode this for now
        start_ix = start
        end_ix = start_ix + samples

        logging.debug(
            f"Extracting audio interval ixs: ({start_ix}, {end_ix}), times: ({stft_frame_to_time(start_ix)}, {stft_frame_to_time(end_ix)})")
        # Check for tracks shorter than the training sample length
        available_samples = self.spect_data.shape[1]
        if available_samples < samples:
            logging.debug("Requested sample length (" + str(available_samples) +
                          ") is longer than the audio length (" + str(samples) + ")")
            raise RangeError("Requested sample length (" + str(available_samples) +
                             ") is longer than the audio length (" + str(samples) + ")")

        # Check for samples that go beyond the end of the track
        if end_ix >= available_samples or start_ix >= available_samples:
            logging.debug("Requested interval (" + str(start_ix) + "," + str(end_ix) +
                          ") goes beyond the end of the track (" + str(available_samples) + ")")
            raise RangeError("Requested interval (" + str(start_ix) + "," + str(
                end_ix) + ") goes beyond the end of the track (" + str(available_samples) + ")")
        # Check for samples that go beyond the start of the track
        if end_ix < 0 or start_ix < 0:
            logging.debug("Requested interval (" + str(end_ix) + "," +
                          str(start_ix) + ") goes beyond the start of the track")
            raise RangeError("Requested interval (" + str(end_ix) + "," +
                             str(start_ix) + ") goes beyond the start of the track")
        # Check for samples that
        # Get the size of the spectrogram data
        # (h, w) = self.data.shape
        # Compute the start in terms of the array
        # There should definitely be a better way of doing this.
        # start_ix = int(math.floor((start / self.length) * w))
        # Compute the end in terms of the length
        # we want it to be consistent across audio file lengths
        # end_ix = start_ix + 1720 # int(math.floor((sample_length / self.length) * w))
        logging.debug("Extracting data in spectrogram interval (" +
                      str(start_ix) + "," + str(end_ix) + ") from " + str(available_samples))
        ret = self.spect_data[:, start_ix:end_ix]
        logging.debug("Returned data shape: " + str(ret.shape))
        return ret


def main(file):
    # create an AudioSpectrogram from 'file'
    print("reading from file: " + file)
    spectf = AudioSpectrogram.from_file_name(file)
    spectf.load_spectrogram()
    spectf.clear()
    spectf.load_spectrogram()

    spectf2 = AudioSpectrogram.from_file_name(file, read_from_cache=False)
    spectf2.load_spectrogram(read_from_cache=False)

    spectf2.plot_spectrogram()

    spectf2.interval(180)


if __name__ == '__main__':
    import argparse
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True,
                        help='Path to load as a spectrogram file')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
