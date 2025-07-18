#!/bin/bash
usage() {
    echo "Usage: $0 -m model -l language -e experiment"
    echo "Options:"
    echo "  -m model      Model name"
    echo "  -d device     CUDA device (optional)"
    echo "  -c concepts   Concepts to evaluate (comma-separated, optional)"
    exit 1
}
# Parse command-line arguments
while getopts "m:d:c:" opt; do
    case "$opt" in
        m) model="$OPTARG" ;; 
        d) device="$OPTARG" ;;
        c) concepts=("$OPTARG") ;;
        *) usage ;;
    esac
done

# echo "Available CUDA devices: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [[ ! -z "$device" ]]; then
    export CUDA_VISIBLE_DEVICES="$device"
    echo "Using CUDA device: $device"
fi

perturbation="uncomment"
if [[ -z "$concepts" ]]; then
    concepts=("comment" "inline" "multiline")
fi

echo "Concepts: ${concepts[*]}"

for concept in "${concepts[@]}"; do
    python -u run.py -m "$model" -p direct --concept "$concept" --perturbation "$perturbation"
done