#!/bin/bash

export BUCKET_NAME="bellson"
export JOB_NAME="bellson_bpm_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/models/$JOB_NAME
export REGION=europe-west1

# gcloud ml-engine local train \
#   --job-dir $JOB_DIR \
#   --module-name trainer.bellson \
#   --package-path ./trainer \
#   --configuration config.yaml \
#   -- \
#   --data-dir ./data/smnp \
#   --ellington-lib ./data/example.el \
#   --job-dir ./logs/ \

gcloud ml-engine jobs submit training $JOB_NAME \
    --module-name trainer.bellson \
    --job-dir $JOB_DIR \
    --runtime-version 1.9 \
    --config config.yaml \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --data-dir ./data/smnp \
    --ellington-lib ./data/example.el \
    --job-dir ./logs/ \