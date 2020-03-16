#!/bin/bash

kg="conceptnet"
kg="visualgenome"

# Get conceptnet triples
python3 src/preprocess.py $kg
echo "${kg} ready"

# Preprocess dataset
python3 src/preprocess.py
echo "dataset ready"

# Start training
python3 -u src/main.py --gpu 0 > run.log
