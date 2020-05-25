import logging
import copy
import random
import numpy as np
from numpy.random import uniform
import gc

from tensorflow.keras.utils import Sequence

from .ellington_library import EllingtonLibrary, Track
from .audio_spectrogram import AudioSpectrogram, RangeError, time_to_stft_frame, stft_frame_to_time
from . import config


class TrackIterator:
    bpm = None
    spect = None
    multiplier = 1
    name = None

    def __init__(self, name, spect, start_cutoff=30, end_cutoff=30, multiplier=1, bpm=None):
        # Set the name for debugging
        self.name = name
        # Load the spectrogram
        self.spect = spect
        # Set the multiplier for yielding tracks
        self.multiplier = multiplier
        # Set the start/end ixs of our allowed sample collection
        track_len = self.spect.audio_length()
        self.start_ix = int(time_to_stft_frame(start_cutoff))
        self.end_ix = int(time_to_stft_frame(
            track_len-end_cutoff) - config.input_time_dim)
        logging.debug(
            f"Clamping samples within times/range ({start_cutoff}, {track_len - end_cutoff}) / ({self.start_ix}, {self.end_ix})")

        if (track_len-end_cutoff) - start_cutoff < 30:
            logging.warn(
                f"Track {self.name} start/end times ({start_cutoff}, {track_len - end_cutoff}) gives less than 30s of data!")
            # If we don't have that much data, we should just use all of it - it's unlikely that there will be slow downs/speedups in a short track.
            self.start_ix = 0
            logging.warn(
                f"Track length {self.spect.audio_length()} may be too short!")
            logging.warn(
                f"Track frames {time_to_stft_frame(self.spect.audio_length())} may be too few!")
            self.end_ix = int(time_to_stft_frame(
                self.spect.audio_length())) - config.input_time_dim
            logging.warn(
                f"Set new start/end times of {self.start_ix}, {self.end_ix}")
        # Set the config values
        self.bpm = bpm

    @classmethod
    def from_filename(cls, filename, start_cutoff=30, end_cutoff=30, multiplier=1):
        return TrackIterator(spect=AudioSpectrogram.from_file_name(filename), name=filename, start_cutoff=start_cutoff, end_cutoff=end_cutoff, multiplier=multiplier, bpm=None)

    @classmethod
    def from_track(cls, track, start_cutoff=30, end_cutoff=30, multiplier=1):
        return TrackIterator(spect=AudioSpectrogram.from_library_track(track), name=track.trackname, start_cutoff=start_cutoff, end_cutoff=end_cutoff,  multiplier=multiplier, bpm=track.bpm)

    def get_random_sample_and_time(self):
        while(True):
            s = int(uniform(self.start_ix, self.end_ix))
            try:
                sample = self.get_sample_at_ix(s)
                return (sample, s)
            except RangeError:
                logging.debug(
                    "Random range was invalid - continuing to try again")

    def get_random_sample(self):
        (sample, _s) = self.get_random_sample_and_time()
        return sample

    def get_sample_at_ix(self, ix):
        logging.debug(f"Yielding data from position {ix}")
        sample = self.spect.interval(ix)
        assert sample.shape == (config.input_freq_dim, config.input_time_dim)
        (w, h) = sample.shape
        return np.reshape(sample, (w, h, 1))

    def get_uniform_batch(self, sample_c=None):
        # Get a uniform set of samples that covers the spectrogram as well as possible.
        # Start off by working out how many samples we'll need
        logging.debug(
            f"end col: {self.end_ix}, start col: {self.start_ix}")
        cols = self.end_ix - self.start_ix
        logging.debug(f"Spectrograph has {cols} columns of data")

        if sample_c is None:
            frames = int(np.ceil(float(cols)/float(config.input_time_dim)))
            logging.debug(
                f"End to end, {frames} frames of data are trivially available")
            assert frames > 0
            frames = frames * self.multiplier
            logging.debug(
                f"Retrieving {frames} frames of data")
        else:
            assert sample_c > 0
            assert sample_c < cols
            frames = sample_c

        indicies = np.linspace(
            self.start_ix, self.end_ix-config.input_time_dim, num=frames)
        samples = np.array([self.get_sample_at_ix(int(np.floor(x)))
                            for x in indicies])
        return samples

        # This assumes that we've been initilised with a bpm

    def get_batch_with_tempos(self):
        samples = self.get_uniform_batch()
        tempos = np.repeat(float(self.bpm)/400.0, samples.shape[0])
        logging.debug(
            f"Samples shape: {samples.shape}, tempos shape: {tempos.shape}")
        return (samples, tempos)

    # def get_batch_with_start_times(self, sample_count):
    #     samples = [self.get_random_sample() for x in range(0, sample_count)]
    #     print(f"Samples type: {type(samples)}")
    #     return (start_times, np.concatenate(np.array(samples)))


class LibraryIterator(Sequence):
    library = None

    def __init__(self, library, start_cutoff=30, end_cutoff=30, multiplier=1):
        # Make a deep copy of the library, so that we can shuffle it.
        self.library = copy.deepcopy(library)
        # Cache config values
        self.start_cutoff = start_cutoff
        self.end_cutoff = end_cutoff
        self.multiplier = multiplier

    def __len__(self):
        """
            Impl of __len__ from tf.keras.utils.Sequence
        """
        return len(self.library.tracks)

    def __getitem__(self, idx):
        assert(idx < len(self.library.tracks))
        # Get track idx from the library
        track = self.library.tracks[idx]
        logging.debug(
            f"idx {idx} / track {track.shortname} / bpm {track.bpm}")
        ti = TrackIterator.from_track(
            track, self.start_cutoff, self.end_cutoff, multiplier=self.multiplier)
        batch = ti.get_batch_with_tempos()  # :: (samples, tempos)
        return batch

    def batch_count(self):
        return len(self.library.tracks)

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        logging.debug("Shuffling library")
        random.shuffle(self.library.tracks)

    def realise_for_validation(self, multiplier=1):
        logging.info(
            f"Realising validation dataset of {len(self.library.tracks)} tracks")
        # samples, bpms :: ([np.arr, np.arr, ...], [np.arr, np.arr])

        samples = []
        bpms = []
        total_samples = 0

        for track in self.library.tracks:
            logging.info(f"Loading track: {track.trackname}")
            (batch, tempo) = TrackIterator.from_track(track, self.start_cutoff,
                                                      self.end_cutoff, multiplier=self.multiplier).get_batch_with_tempos()
            (s, _, _, _) = batch.shape
            total_samples += s
            samples.append(batch)
            bpms.append(tempo)

        logging.info(f"Loaded {total_samples} total samples.")

        arr_samples = np.concatenate(samples)
        arr_bpms = np.concatenate(bpms)

        del samples
        del bpms

        logging.debug(f"Sample(s) shape: {arr_samples.shape}")
        logging.debug(f"Bpm(s) shape: {arr_bpms.shape}")

        return (arr_samples, arr_bpms)
