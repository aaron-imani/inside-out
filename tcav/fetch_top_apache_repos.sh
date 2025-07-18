#!/bin/bash

# Top 10 Java Apache projects by stars
project_names=("dubbo" "kafka" "flink" "skywalking" "rocketmq" "shardingsphere" "hadoop" "pulsar" "doris")
mkdir -p repos
cd repos

for project in "${project_names[@]}"; do
    echo "Fetching ${project}..."
    git clone --depth=1 --filter=blob:none --sparse https://github.com/apache/${project}.git
    cd ${project}
    git sparse-checkout set '**/*.java'
    # if not successful, try the main branch
    # if [ $? -ne 0 ]; then
    #     git pull --depth=1 origin main
    # fi
    cd ..
done