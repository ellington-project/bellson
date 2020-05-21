#!/bin/bash

# Run this script from the top level of bellson 
# This script visualises the accuracy of the network on a set of songs. 

export NOW=$(date +%F_%H.%M)
export PLOTD="/mnt/bigboi/visualisation_plots/$NOW"
mkdir -p $PLOTD

echo "Generating visualisations in directory $PLOTD"

export TF_FORCE_GPU_ALLOW_GROWTH=true


# Run visualisation
python3 -m bellson.apps.visual.visualise \
    --ellington-lib=/mnt/bigboi/library.json \
    --cache-dir=/mnt/bigboi/training_cache/ \
    --modelfile=/mnt/bigboi/training_runs/current/latest-model.h5 \
    --plotd=$PLOTD