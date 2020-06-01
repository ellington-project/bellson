#!/bin/bash 

# Call from the root of the bellson project. 
# The folder storing model files: 
modeld=/mnt/bigboi/training_runs/current
models=$(ls -t $modeld | head | grep "model-epoch")
samplec=500
tracks="oocj-169bpm.flac
aom-156.m4a"
for model in $models; do
    echo "Model $model ::"
    for track in $tracks; do 
        result=$(python3 -m bellson.apps.tf.infer --audio-file ~/Music/$track --samples $samplec --model-file $modeld/$model 2>/dev/null | tail -n 1)
        echo "      - $track == $result "    
    done


done

