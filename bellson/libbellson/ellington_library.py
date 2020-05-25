# Utility class + methods for parsing an ellington library

# Not a 1:1 correspondence between this and the Rust definition - we don't need all the information that the library contains right now.

from __future__ import print_function
from multiprocessing import Pool
from pprint import pprint, pformat

import hashlib
import json
import os
import math
import logging
import re


class Track:
    bpm = None
    filename = None
    trackname = None
    digest = None

    # Variants for training
    variant = None

    # From other estimators
    naive_bpm = None
    old_bellson_bpm = None

    @classmethod
    def from_json(cls, json):
        bpm = None
        filename = None
        trackname = None
        naive_bpm = None,
        old_bellson_bpm = None

        # get the track filename
        try:
            filename = json['location']
        except:
            logging.debug("Track has no location information, ignoring.")
            return None

        # get the bpm (non-optional for training tracks)
        try:
            bpm = float(json['metadata']['bpm'])
            if bpm is None:
                return None
            if bpm < 100:
                logging.debug("Track is too slow to use for training.")
                return None
        except:
            logging.debug(
                "No bpm member in json metadata, not using this track for training")
            return None

        # Get the name of the track
        try:
            trackname = json['metadata']['name']

        except:
            logging.error("Failed to get (non-optional) name member of json")
            return None

        # Optionally get data from previous inference attempts
        try:
            naive_bpm = json["eldata"]["algs"]["Naive"]["Bpm"]
        except:
            logging.debug("Failed to get (optional) naive bpm from json")

        try:
            old_bellson_bpm = json["eldata"]["algs"]["Bellson"]["Bpm"]
        except:
            logging.debug("Failed to get (optional) old Bellson bpm from json")

        return Track(bpm, filename, trackname, 1.0, naive_bpm, old_bellson_bpm)

    def __init__(self, bpm, filename, trackname, variant=1.0, naive_bpm=None, old_bellson_bpm=None):
        self.bpm = bpm
        self.filename = filename
        self.trackname = trackname
        self.variant = variant
        self.digest = hashlib.sha256(filename.encode(
            'utf-8')).hexdigest() + "-" + str(variant)
        self.naive_bpm = naive_bpm
        self.old_bellson_bpm = old_bellson_bpm
        self.shortname = re.sub(
            " \(.*\)", "", re.sub("\[.*\] ", "", trackname))

    def as_variants(self, variants):
        return [Track(
            bpm=self.bpm*variant,
            filename=self.filename,
            trackname=self.trackname + "-" + str(variant),
            variant=variant,
            naive_bpm=self.naive_bpm,
            old_bellson_bpm=self.old_bellson_bpm) for variant in variants]

    def __str__(self):
        return "T["+str(self.filename) + "," + str(self.trackname) + "," + str(self.bpm) + "]"

    def __repr__(self):
        return "T["+str(self.filename) + "," + str(self.trackname) + "," + str(self.bpm) + "]"

    def librosa_tempo(self):
        import librosa

        SAMPLE_RATE = 44100

        (y, sr) = librosa.load(self.filename,
                               sr=SAMPLE_RATE, res_type='kaiser_fast')

        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr)

        t = tempo.item(0)
        return t


class EllingtonLibrary:
    tracks = []

    @classmethod
    def from_file(cls, filename, maxsize=None):
        with open(filename, "r") as f:
            json_data = f.read()
            return EllingtonLibrary.from_json(json_data, maxsize)

    @classmethod
    def from_json(cls, jsonstr, maxsize=None):
        json_data = json.loads(jsonstr)
        t = list(filter(lambda t: t is not None,
                        map(Track.from_json, json_data['tracks'])))

        return EllingtonLibrary(t, maxsize)

    def __init__(self, t, maxsize=None):
        if maxsize is not None:
            self.tracks = t[0:maxsize]
        else:
            self.tracks = t

    def augment_library(self, variants, validate_lengths=False, maxsize=None):
        track_variants = []
        logging.info(f"Generating tempo variants: {variants}")
        for track in self.tracks:
            logging.debug(f"Generating variants for: {track} ")
            # (optionally) validate the lengths to check that the tracks are long enough.
            if validate_lengths:
                import librosa
                original_length = librosa.get_duration(filename=track.filename)
                logging.info(
                    f"Validating variants for len/track {original_length} / {track.shortname}")
                for variant in track.as_variants(variants):
                    new_length = original_length * variant.variant
                    if new_length > 45:
                        track_variants.append(variant)
                    else:
                        logging.warn(
                            f"Track variant {track.shortname}, {variant.variant} did not yield enough data (45 s), so skipping extending library with it.")
            else:
                track_variants.extend(track.as_variants(variants))
        self.tracks.extend(track_variants)

        if maxsize is not None:
            self.tracks = self.tracks[0:maxsize]
        else:
            self.tracks = self.tracks

    def split_training_validation(self, ratio=10):
        n = ratio
        assert n > 0
        validation_tracks = [item for index, item in enumerate(
            self.tracks) if (index) % n == 0]
        training_tracks = [item for index, item in enumerate(
            self.tracks) if (index) % n != 0]
        (validation_size, training_size) = (
            len(validation_tracks), len(training_tracks))

        logging.debug(f"Validation dataset size (tracks): {validation_size}")
        logging.debug(f"Training dataset size (tracks): {training_size}")

        return (EllingtonLibrary(training_tracks), EllingtonLibrary(validation_tracks))

    def len(self):
        return len(self.tracks)

    def __str__(self):
        return str(self.tracks)

    def __repr__(self):
        return pformat(self.tracks)


def main(file):
    el = EllingtonLibrary.from_file(file)
    el.augment_library([0.75, 0.8, 0.85,
                        0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25])
    for track in el.tracks:
        logging.info(f"Track: {track}")


if __name__ == '__main__':
    import logging
    import argparse
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True,
                        help='Path to load the ellington library from')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
