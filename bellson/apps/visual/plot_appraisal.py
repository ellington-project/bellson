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

import re
import operator


def replace(s):
    m = re.search("model-epoch-(\d+)-.*", s)
    epoch = int(m.group(1))
    return epoch


def main(plotd, data_files):
    sns.set(color_codes=True)
    sns.set_style(style='white')
    sns.set_palette("deep")

    fig, ax = plt.subplots(figsize=(20, 20))

    frames = []

    for file in data_files:
        logging.info(f"Loading data from file: {file}")
        data = pd.read_csv(file, delimiter="|", header=0)
        data["model"] = data["model"].apply(replace)
        if data["model"][0] <= 100:
            frames.append(data)

    data = pd.concat(frames)

    g = sns.FacetGrid(data, col="model", col_wrap=10)

    g.map(sns.lineplot, "expected_bpm", "predicted_bpm")

    for ax in g.axes.flat:
        ax.set_frame_on(False)
        ax.plot((100, 300), (100, 300), c=".2", ls="--")

    plt.savefig("plots/tempo_distribution.png")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--plotd', required=False, default="plots",
                        help='A directory in which to store generated plots.')

    parser.add_argument("--data-files", required=True, nargs='+',
                        help='Files of data to plot of the form "model|expected|predicted"')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
