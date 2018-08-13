# Utility class + methods for parsing an ellington library

# Not a 1:1 correspondence between this and the Rust definition - we don't need all the information that the library contains right now. 

from __future__ import print_function
from multiprocessing import Pool
from pprint import pprint, pformat

import hashlib
import json
import os
import math
from pathlib import Path

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
            if bpm is not None and bpm is not 0 and trackname is not None:
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
    def from_file(cls, filename):
        with open(filename) as f:
            json_data = json.load(f)
            return EllingtonLibrary(json_data)

    def __init__(self, json):
        self.tracks = list(filter(lambda t: t is not None,
                                  map(Track.from_json, json['tracks'])))

    def __str__(self):
        return str(self.tracks)

    def __repr__(self):
        return pformat(self.tracks)
