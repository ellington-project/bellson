#!/bin/bash

# Change to the "bellson" folder
pushd $(dirname $0)
bellson=$(pwd -P)

# fix the model for now. 
model="$bellson/models/bpm-model.h5"
echo $model

# get an audio file
audiof=$1
echo $audiof

python3 -m bellson.apps.infer --model "$model" --audiofile "$audiof" --inc 1

popd
