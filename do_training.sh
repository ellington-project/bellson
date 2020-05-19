#!/bin/bash

export NOW=$(date +%F_%H.%M)
export LOGD="/mnt/bigboi/training_runs/$NOW"
mkdir -p $LOGD

echo "Running training in directory $LOGD"
rm -f "/mnt/bigboi/training_runs/current"
ln -s $LOGD "/mnt/bigboi/training_runs/current"

export TF_FORCE_GPU_ALLOW_GROWTH=true 

python3 bellson/bellson_train.py \
    --ellington-lib=/mnt/bigboi/library.json \
    --job-dir=$LOGD \
    --cache-dir=/mnt/bigboi/training_cache/