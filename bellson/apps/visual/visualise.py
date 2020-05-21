#!/usr/bin/env python3
import argparse
import logging
import gc
import objgraph
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ...libbellson.ellington_library import EllingtonLibrary, Track
from ...libbellson.library_iterator import LibraryIterator, TrackIterator
from ...libbellson.model import load_model
from ...libbellson import config

import re
import operator


def plot_inference_accuracy(tracks, filename, model):
    results_dict = {}
    tracks.sort(key=operator.attrgetter('bpm'))
    trackc = len(tracks)
    i = 0
    for track in tracks:
        track_name = re.sub("\[.*\] ", "", track.trackname)
        track_name = re.sub(" \(.*\)", "", track_name)
        logging.info(
            f"Predicting track {i}/{trackc} - {track_name} (bpm: {track.bpm})")
        i = i + 1

        ti = TrackIterator.from_track(track)
        logging.info("Loading audio samples")
        audio_samples = ti.get_uniform_batch(sample_c=128)

        logging.info("Running network")
        results = model.predict_on_batch(
            audio_samples).numpy().flatten().tolist()

        predictions = list(map(lambda s: s * 400, results))

        # logging.info("Running naive librosa method")
        # librosa_tempo = track.librosa_tempo()

        results_dict[track_name] = {
            'bpm': track.bpm,
            # 'librosa': librosa_tempo,
            'predictions': predictions}

    logging.info("Transforming results dictionary into list...")
    aggregate_results = []
    for track, data in results_dict.items():
        predictions = data['predictions']
        for prediction in predictions:
            aggregate_results.append(
                {'track': track, 'prediction': prediction})

    logging.info("Transforming results list into dataframe...")
    results_df = pd.DataFrame(aggregate_results)

    logging.info("Generating plot.")

    def custom_kedplot(predictions,  **kwargs):
        name = kwargs['label']
        bpm = results_dict[name]['bpm']
        # librosa = results_dict[name]['librosa']
        logging.info(f"Plotting: {name}/{bpm}")
        sns_plot = sns.kdeplot(predictions, shade=True)
        # sns_plot.axes.get_yaxis().set_visible(False)
        sns_plot.set_frame_on(False)

        plt.axvline(x=bpm, color=sns.color_palette()[1])
        # plt.axvline(x=librosa, color=sns.color_palette()[2])

    g = sns.FacetGrid(results_df, col="track", hue="track", height=3, aspect=1.5,
                      col_wrap=int(math.ceil(math.sqrt(len(tracks)))))

    g.map(custom_kedplot, "prediction").set_titles(
        "{col_name}").set_axis_labels("Predicted BPM")
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    g.savefig(filename)


def main(cache_dir="/tmp", ellington_lib="data/example.el",  modelfile="nofile", plotd="plotd"):
    config.cache_directory = cache_dir
    sns.set(color_codes=True)
    sns.set_style(style='white')
    sns.set_palette("deep")

    logging.info("Loading model")
    model = load_model(modelfile)

    logging.info("Loaded model")

    # Set up the data input etc.
    overall_library = EllingtonLibrary.from_file(ellington_lib)
    (train_lib, valid_lib) = overall_library.split_training_validation()

    plot_inference_accuracy(valid_lib.tracks, plotd +
                            "/validation_accuracy.png", model)
    # Too big - don't do this!
    # plot_inference_accuracy(train_lib.tracks, "training_accuracy.png", model)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True,
                        help='Path to cache directory, for pre-compiled histograms')
    parser.add_argument('--ellington-lib', required=True,
                        help='The ellington library from which to read track names and BPMs')
    parser.add_argument("--modelfile", required=True,
                        help='Model to use for inference')
    parser.add_argument("--plotd", required=True,
                        help='Directory to write plots to')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
