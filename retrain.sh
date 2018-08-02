#!/bin/bash
python -m scripts.retrain \
    --bottleneck_dir=tf_files/bottlenecks/ \
    --model_dir=tf_files/models/ \
    --how_many_training_steps=10000 \
    --summaries_dir=tf_files/training_summaries/"inception_v3" \
    --output_graph=tf_files/retrained_graph.pb \
    --output_labels=tf_files/retrained_labels.txt \
    --architecture="inception_v3" \
    --image_dir=/home/adam/personal/bellson/tmp/spect/


