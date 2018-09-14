# Utility class + methods for parsing an ellington library

# Not a 1:1 correspondence between this and the Rust definition - we don't need all the information that the library contains right now. 

from __future__ import print_function
from multiprocessing import Pool
from pprint import pprint, pformat

import hashlib
import json
import os
import math
from .path import Path

from tensorflow.python.lib.io import file_io

class LibEntry:
    bpm = None
    filename = None
    trackname = None

    @classmethod
    def from_json(cls, json):
        metadata = json['metadata']
        filename = json['location']
        if metadata is not None and filename is not None:
            bpm = metadata['bpm']
            trackname = metadata['name']
            if bpm is not None and bpm > 100 and trackname is not None:
                return LibEntry(bpm, filename, trackname)

        return None

    def __init__(self, bpm, filename, trackname):
        self.bpm = bpm
        self.filename = Path(filename)
        self.trackname = trackname


    def __str__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"

    def __repr__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"


class EllingtonLibrary:
    tracks = []

    @classmethod
    def from_file(cls, filename, maxsize=None):
        with file_io.FileIO(filename, "r") as f:
            json_data = json.load(f)
            return EllingtonLibrary(json_data, maxsize)

    @classmethod 
    def from_json(cls, jsonstr, maxsize=None): 
        json_data = json.loads(jsonstr)
        return EllingtonLibrary(json_data, maxsize)

    def __init__(self, json, maxsize=None):
        t = list(filter(lambda t: t is not None,
                                  map(LibEntry.from_json, json['tracks'])))
        if maxsize is not None: 
            self.tracks = t[0:maxsize]
        else:
            self.tracks = t

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
    parser.add_argument('--file', required=True, help='Path to load the ellington library from')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)

