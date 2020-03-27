#! /bin/bash

cfg="cfg/socialiqa-train-dev.yaml"
for dataset in "socialiqa-train-dev" #"alphanli" #"physicaliqa-train-dev" # "socialiqa-train-dev"
do
	python -m mowgli --dataset $dataset --output output/ --config $cfg 
done
