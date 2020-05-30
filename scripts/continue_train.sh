#!/bin/bash

# Run this script from the top level of bellson 
# This script trains a network on some dataset, continuing on from an existing (partial) training run . 

export LOGD="/mnt/bigboi/training_runs/current"

echo "Running training in directory $LOGD"

export TF_FORCE_GPU_ALLOW_GROWTH=true

# Start tensorboard at $LOGD
tensorboard --logdir $LOGD --bind_all & 

# Run training - do so in a loop in case our training fails with OOM. 
until python3 -m bellson.apps.tf.train \
    --ellington-lib=/mnt/bigboi/library.json \
    --job-dir=$LOGD \
    --cache-dir=/mnt/bigboi/training_cache/; do 
    echo "Training process either finished or was killed with (probably) OOM."
    echo "Restarting in 5 minutes."
    sleep 300
done