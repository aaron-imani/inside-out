#!/bin/bash

concepts=("comment" "inline"  "javadoc" "multiline")
model_name=$1

# Check if the file exists
# if [ -f "rq2_results/${model_name}_layers_accuracies.csv" ]; then
#     rm rq2_results/${model_name}_layers_accuracies.csv
#     echo "Removed existing file: rq2_results/${model_name}_layers_accuracies.csv"
# fi

for dataset_name in "${concepts[@]}"; do
    python -u train_cav.py $model_name $dataset_name -r 10
    sleep 5
done