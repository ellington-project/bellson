#!/bin/bash 
set -e
here=$(pwd)
mkdir -p logs 
mkdir -p cache 
docker build . -t trainimg 
docker run -it --gpus all \
    -v /mnt/bigboi:/mnt/bigboi \
    -v $here/bellson:/bellson/bellson \
     trainimg 