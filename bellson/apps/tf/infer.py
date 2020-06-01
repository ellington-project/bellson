#!/usr/bin/env python3
from ...libbellson.models import list_distributed_models
import argparse
import logging
import objgraph
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# from bellson.audio import Track, CacheLevel, TrackIterator, RangeError


def main(model_file, audio_file, cache_dir, samples):
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    from ...libbellson.library_iterator import TrackIterator
    from ...libbellson.model import load_model

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    start_time = time.time()
    logging.info("Started infererence tool")
    # Create the model, print info
    logging.info("Loading model")
    model = load_model(model_file)

    logging.info("Loaded model")

    # create a track from the audio_file file path, and load spectrogram data
    # Don't cache, as this just takes up file space
    track_iterator = TrackIterator.from_filename(audio_file)
    logging.info(f"Loaded track from path {audio_file}")

    logging.info("Reading samples from track")
    samples = track_iterator.get_uniform_batch(sample_c=int(samples))

    logging.info("Predicting batch")
    results = model.predict_on_batch(samples)
    if isinstance(results, tf.python.framework.ops.EagerTensor):
        results = results.numpy().flatten().tolist()
    else:
        results = results.flatten().tolist()

    mean = np.mean(results)  # * 400
    stddev = np.std(results)  # * 400

    logging.debug("Results: [{}]".format("\n ".join(
        ['%.2f' % r for r in results])))
    # ['%.2f' % (r * 400) for r in results])))
    logging.info(f"Mean: {mean:.3f}")
    logging.info(f"Stddev: {stddev:.4f}")

    logging.info(f"Inference took {(time.time() - start_time)}s")
    print(str(int(np.round(mean))))


def entrypoint():
    models = list_distributed_models()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    if len(models) > 0:
        model = models[0]
        parser.add_argument('--model-file', required=False, default=model,
                            help=f'The model to use for inference, default {model}')
    else:
        parser.add_argument('--model-file', required=True,
                            help='The model to use for inference')
    parser.add_argument('--audio-file', required=True,
                        help='The audio file to analyse')
    parser.add_argument('--cache-dir', required=False, default="/tmp",
                        help='Path to cache directory, for pre-compiled histograms')
    parser.add_argument('--samples', required=False, type=int, default=100,
                        help='Number of samples to stract from track and perform inference on')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)


if __name__ == '__main__':
    entrypoint()
