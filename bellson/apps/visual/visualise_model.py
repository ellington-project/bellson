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

from ...libbellson.ellington_library import EllingtonLibrary, Track
from ...libbellson import config
from ...libbellson.model import load_model

import re
import operator


def main(model):

    m = load_model(model)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='The model to visualise/example')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
