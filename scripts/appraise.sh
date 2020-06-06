#!/bin/bash

# Run this script from the top level of bellson 
# This script apprases a directory of models, outputting data in bellson/data

export TF_FORCE_GPU_ALLOW_GROWTH=true

# Call from the root of the bellson project. 
modeld=$(realpath /mnt/bigboi/training_runs/current)
dname=$(basename $modeld)
datad="/mnt/bigboi/appraisal_data/$dname/"
mkdir -p $datad

models=$(ls -r $modeld | grep "hdf5")

for m in $models; do 
    echo "Appraising model: $m"
    python3 -m bellson.apps.tf.appraise \
    --cache-dir /mnt/bigboi/training_cache/ \
    --ellington-lib /mnt/bigboi/library.json \
    --sample-count 10 \
    --resultd $datad \
    --models $modeld/$m
done
