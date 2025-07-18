#!/bin/bash

concepts=("comment" "inline" "javadoc" "multiline")
train_sizes=(0.01 0.05 0.1 0.25 0.5)
model_name=$1

for dataset_name in "${concepts[@]}"; do
  for train_size in "${train_sizes[@]}"; do
    python -u train_cav.py $model_name $dataset_name -t $train_size -r 1
    sleep 5
  done
done