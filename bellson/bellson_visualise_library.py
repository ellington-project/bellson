#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common.ellington_library import EllingtonLibrary, Track
import common.config as config

import re
import operator


def main(ellington_lib="data/example.el",  plotd="plots"):
    sns.set(color_codes=True)
    sns.set_style(style='white')
    sns.set_palette("deep")

    library = EllingtonLibrary.from_file(ellington_lib)

    fig, ax = plt.subplots()

    for member in ['naive_bpm', 'bpm', 'old_bellson_bpm']:
        tempos = list(map(lambda t: getattr(t, member), library.tracks))
        plot = sns.kdeplot(tempos, label=member, ax=ax,
                           shade=False, legend=True)
        plot.set_frame_on(False)

    plt.savefig("plots/tempo_distribution.png")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    parser.add_argument('--plotd', required=True,
                        help='A directory in which to store generated plots.')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
