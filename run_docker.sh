#!/bin/bash 
set -e

if [ -z "$1" ]; then 
    echo "This script requries the name of a script, e.g. 'do_training.sh, that exists in 'scripts/'"
    echo "This will be run in the docker container, i.e. './scripts/do_training.sh'"
    exit -1 
fi 
script=./scripts/$1
if [ -f "$script" ]; then 
    echo "Running script: $script"
else 
    echo "Error, script '$script' not found. Failing."
    exit -2
fi

here=$(pwd)
mkdir -p logs 
mkdir -p cache 
docker build -f Dockerfile . -t trainimg 
docker run -it --gpus all \
    -v /mnt/bigboi:/mnt/bigboi \
    -v $here/bellson:/bellson/bellson \
    -v $here/scripts:/bellson/scripts \
    -p 6007:6006 \
    trainimg \
    $script