#!/usr/bin/env python3
import logging
import argparse

from multiprocessing import Pool

from trainer.ellington_library import EllingtonLibrary, Track
from trainer.audio import Audio

ddir = "data/smnp"


def proc(tp):
    ix = tp[0]
    track = tp[1]

    print("Track: " + str(track.trackname) + " " + str(ix))
    audiotrack = Audio(track)
    audiotrack.load()
    audiotrack.save_spectrogram(ddir)


def main(data_dir="data/smnp/", ellington_lib="data/example.el"):
    ddir = data_dir
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=print)
    el = EllingtonLibrary.from_file("data/small.el")

    print("Total tracks: " + str(len(el.tracks)))
    tracks = list(enumerate(el.tracks))

    pool = Pool(10)
    pool.map(proc, tracks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Path to save training data, in the form of compressed numpy arrays')
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
