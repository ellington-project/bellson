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


def main(plotd, data_files):
    sns.set(color_codes=True)
    sns.set_style(style='white')
    sns.set_palette("deep")

    fig, ax = plt.subplots(figsize=(20, 20))

    for file in data_files:
        data = pd.read_csv(file, delimiter="|", header=0)

        sns.lineplot(x="expected_bpm", y="predicted_bpm",
                     hue="model", data=data, ax=ax)

    sns.lineplot(x=[100, 300], y=[100, 300], ax=ax,
                 label="expected", color="gray")

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
