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
from tensorflow.python.lib.io import file_io


class Track:
    bpm = None
    filename = None
    trackname = None
    digest = None

    @classmethod
    def from_json(cls, json):
        metadata = json['metadata']
        filename = json['location']
        if metadata is not None and filename is not None:
            bpm = metadata['bpm']
            trackname = metadata['name']
            if bpm is not None and bpm > 100 and trackname is not None:
                return Track(bpm, filename, trackname)

        return None

    def __init__(self, bpm, filename, trackname):
        self.bpm = bpm
        self.filename = filename
        self.trackname = trackname
        self.digest = hashlib.sha256(trackname.encode('utf-8')).hexdigest()

    def __str__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"

    def __repr__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"


class EllingtonLibrary:
    tracks = []

    @classmethod
    def from_file(cls, filename, maxsize=None):
        with file_io.FileIO(filename, "r") as f:
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

    def split_training_validation(self, ratio=10):
        n = ratio-1
        assert n > 0
        validation_tracks = [item for index, item in enumerate(
            self.tracks) if (index + 1) % n == 0]
        training_tracks = [item for index, item in enumerate(
            self.tracks) if (index + 1) % n != 0]
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
    print(str(el))


if __name__ == '__main__':
    import logging
    import argparse
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True,
                        help='Path to load the ellington library from')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
