#!/usr/bin/env python3
import logging
import argparse

from multiprocessing import Pool

from ...libbellson.ellington_library import EllingtonLibrary, Track
from ...libbellson.audio_spectrogram import AudioSpectrogram
from ...libbellson import config

total = 0


def proc(tp):
    global total
    ix = tp[0]
    track = tp[1]

    print("Track: [" + str(ix) + "/" + str(total) + "] " +
          str(track.trackname))
    audiotrack = AudioSpectrogram.from_library_track(track)


def main(cache_dir="data/smnp/", ellington_lib="data/example.el"):
    config.cache_directory = cache_dir
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=print)
    el = EllingtonLibrary.from_file(ellington_lib)

    global total
    total = len(el.tracks)
    print("Total tracks: " + str(total))
    tracks = list(enumerate(el.tracks))

    pool = Pool(10)
    pool.map(proc, tracks)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True,
                        help='Path to cache directory, for pre-compiled histograms')
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
