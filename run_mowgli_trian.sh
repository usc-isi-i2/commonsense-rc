#! /bin/bash

mkdir -p output

#rm output/*
#rm checkpoint/*.mdl

cfg="cfg/trian.yaml"
#cfg="cfg/default.yaml"

#datasets="hellaswag-train-dev" #"alphanli" #"physicaliqa-train-dev" # "socialiqa-train-dev"
datasets="se2018t11"

for dataset in $datasets 
do
	python -m mowgli --dataset $dataset --output output/conceptnet-se2018t11 --config $cfg 
done
