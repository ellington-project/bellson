#!/bin/bash

export BUCKET_NAME="bellson"
export JOB_NAME="bellson_bpm_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/models/$JOB_NAME
export DATA_DIR=gs://$BUCKET_NAME/data/sparrays
export LIBRARY=gs://$BUCKET_NAME/data/example.el
export REGION=europe-west1

# gcloud ml-engine local train \
#   --job-dir $JOB_DIR \
#   --module-name trainer.bellson \
#   --package-path ./trainer \
#   --configuration config.yaml \
#   -- \
#   --data-dir $DATA_DIR \
#   --ellington-lib $LIBRARY \
#   --job-dir $JOB_DIR \

gcloud ml-engine jobs submit training $JOB_NAME \
    --module-name trainer.bellson \
    --job-dir $JOB_DIR \
    --runtime-version 1.9 \
    --config config.yaml \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --data-dir $DATA_DIR \
    --ellington-lib $LIBRARY \
    --job-dir $JOB_DIR \

gcloud ml-engine jobs stream-logs $JOB_NAME