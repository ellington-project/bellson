#!/bin/bash

# Run this script from the top level of bellson 
# This script trains a network from scratch on some dataset. 

export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run training
python3 -m bellson.apps.tf.evaluate \
    --ellington-lib=/mnt/bigboi/library.json \
    --cache-dir=/mnt/bigboi/training_cache/ \
    --models $(echo "/mnt/bigboi/training_runs/best/"*)