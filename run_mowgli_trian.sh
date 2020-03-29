#! /bin/bash

rm data/*.log
rm data/*vocab
rm data/*.json
rm checkpoint/*.mdl

cfg="cfg/socialiqa-train-dev.yaml"
for dataset in "socialiqa-train-dev" #"alphanli" #"physicaliqa-train-dev" # "socialiqa-train-dev"
do
	python -m mowgli --dataset $dataset --output output/ --config $cfg 
done
