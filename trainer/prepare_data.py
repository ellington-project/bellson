#!/usr/bin/env python3
import logging

from multiprocessing import Pool

from trainer.ellington_library import EllingtonLibrary, Track
from trainer.audio import Audio


def proc(tp):
    ix = tp[0]
    track = tp[1]

    print("Track: " + str(track.trackname) + " " + str(ix))
    audiotrack = Audio(track)
    audiotrack.load()
    audiotrack.save_spectrogram("data/small/")

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    el = EllingtonLibrary.from_file("data/small.el")

    print("Total tracks: " + str(len(el.tracks)))
    tracks = list(enumerate(el.tracks))

    pool = Pool(10)
    pool.map(proc, tracks)

if __name__ == '__main__':
    main()
