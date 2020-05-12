import logging
import copy
import random
# from numpy import array
import numpy as np
from numpy.random import uniform
import gc

from tensorflow.keras.utils import Sequence

from common.ellington_library import EllingtonLibrary, Track
from common.audio_spectrogram import AudioSpectrogram, RangeError, time_to_stft_frame, stft_frame_to_time

SAMPLE_W = 256
SAMPLE_H = 1720


class SpectIterator:
    spect = None

    def __init__(self, filename, start_cutoff=30, end_cutoff=30, samples=10):
        # Load the spectrogram
        self.spect = AudioSpectrogram.from_file_name(filename)
        # Set the start/end ixs of our allowed sample collection
        track_len = self.spect.audio_length()
        self.start_ix = int(time_to_stft_frame(start_cutoff))
        self.end_ix = int(time_to_stft_frame(track_len-end_cutoff) - SAMPLE_H)
        logging.debug(
            f"Clamping samples within range ({self.start_ix}, {self.end_ix})")
        # Set the config values
        self.samples = samples

    def get_sample(self):
        while(True):
            s = int(uniform(self.start_ix, self.end_ix))
            try:
                logging.debug(f"Yielding data from position {s}")
                sample = self.spect.interval(s)
                if sample.shape == (SAMPLE_W, SAMPLE_H):
                    (w, h) = sample.shape
                    sample = np.reshape(sample, (w, h, 1))
                    return (s, sample)
            except RangeError:
                logging.debug(
                    "Random range was invalid - continuing to try again")

    def get_batch(self):
        (start_times, samples) = zip(
            *[self.get_sample() for x in range(0, self.samples)])
        return (start_times, np.array(samples))


class TrackIterator:
    track = None
    spect = None

    def __init__(self, track, start_cutoff=30, end_cutoff=30, samples=10):
        # Set the values, and load the spectrogram
        self.track = track
        self.spect = AudioSpectrogram.from_library_track(track)
        # Set the start/end ixs of our allowed sample collection
        track_len = self.spect.audio_length()
        self.start_ix = int(time_to_stft_frame(start_cutoff))
        self.end_ix = int(time_to_stft_frame(track_len-end_cutoff) - SAMPLE_H)
        logging.debug(
            f"Clamping samples within range ({self.start_ix}, {self.end_ix})")
        # Set the config values
        self.samples = samples

    def get_sample(self):
        while(True):
            s = int(uniform(self.start_ix, self.end_ix))
            try:
                logging.debug(f"Yielding data from position {s}")
                sample = self.spect.interval(s)
                assert(sample.shape == (SAMPLE_W, SAMPLE_H))
                (w, h) = sample.shape
                sample = np.reshape(sample, (w, h, 1))
                return sample
            except RangeError:
                logging.debug(
                    "Random range was invalid - continuing to try again")

    def get_batch(self):
        samples = np.array([self.get_sample() for x in range(0, self.samples)])
        tempos = np.repeat(float(self.track.bpm)/400.0, self.samples)
        logging.debug(
            f"Samples shape: {samples.shape}, tempos shape: {tempos.shape}")
        return (samples, tempos)


class LibraryIterator(Sequence):
    library = None

    def __init__(self, library, start=60, end=180, batch_size=32):
        # Make a deep copy of the library, so that we can shuffle it.
        self.library = copy.deepcopy(library)
        # Cache config values
        self.start = start
        self.end = end
        self.batch_size = batch_size

    def __len__(self):
        """
            Impl of __len__ from tf.keras.utils.Sequence
        """
        return len(self.library.tracks)

    def __getitem__(self, idx):
        assert(idx < len(self.library.tracks))
        # Get track idx from the library
        track = self.library.tracks[idx]
        logging.debug(f"Batch {idx} / track {track}")
        ti = TrackIterator(track, self.start, self.end, self.batch_size)
        batch = ti.get_batch()  # :: (samples, tempos)
        return batch

    def batch_count(self):
        return len(self.library.tracks)

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        logging.debug("Shuffling library")
        random.shuffle(self.library.tracks)

    def realise_for_validation(self):
        logging.info(
            f"Realising validation dataset of {len(self.library.tracks)} tracks")
        # samples, bpms :: ([np.arr, np.arr, ...], [np.arr, np.arr])
        samples, bpms = zip(*[TrackIterator(
            track, self.start, self.end, self.batch_size).get_batch() for track in self.library.tracks])

        arr_samples, arr_bpms = np.array(samples).reshape((self.batch_count()*self.batch_size, SAMPLE_W, SAMPLE_H, 1)), np.array(
            bpms).reshape(self.batch_count() * self.batch_size)

        del samples
        del bpms

        logging.debug(f"Sample(s) shape: {arr_samples.shape}")
        logging.debug(f"Bpm(s) shape: {arr_bpms.shape}")

        return (arr_samples, arr_bpms)

    # def iter(self):
    #     # Go across <self.iterations> iterations
    #     for i in range(0, self.iterations):
    #         # Start by shuffling the library
    #         self.shuffle()
    #         # Iterate over the tracks, and get 20 random samples.
    #         for t in self.library.tracks:
    #             print("Yielding spectrogram data for " + t.trackname)
    #             ti = TrackIterator(t, self.start,
    #                                self.end, self.length, self.samples)
    #             # Generate <samples> random samples from the track, and yield them
    #             for s in ti.iter():
    #                 yield s
    #             del ti

    # def batch(self):
    #     # Yield an iterator over batches of samples
    #     # Iterate over the iterator:
    #     ix = 0
    #     inputs = []
    #     targets = []
    #     for s in self.iter():
    #         target = float(s[0]) / 400.0
    #         targets.append(target)

    #         inp = s[1]
    #         (w, h) = inp.shape
    #         maxv = np.max(np.abs(inp))
    #         data = np.reshape(inp, (w, h, 1)) / maxv
    #         inputs.append(data)

    #         ix = ix + 1

    #         if(ix == self.batchsize):
    #             inputs_arr = np.stack(inputs, axis=0)
    #             targets_arr = np.stack(targets, axis=0)

    #             logging.info("Yielding an array of " +
    #                          str(inputs_arr.shape) + " samples")

    #             yield inputs_arr, targets_arr

    #             del inputs
    #             del targets
    #             gc.collect()
    #             inputs = []
    #             targets = []
    #             ix = 0
