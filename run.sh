#!/bin/bash

# Get conceptnet triples
python3 src/preprocess.py conceptnet
echo "conceptnet ready"

# Preprocess dataset
python3 src/preprocess.py
echo "dataset ready"

# Start training
python3 -u src/main.py --gpu 0 > run.log
