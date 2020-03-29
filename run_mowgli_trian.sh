#! /bin/bash

mkdir -p output

#rm output/*.*
rm checkpoint/*.mdl

cfg="cfg/trian.yaml"
#cfg="cfg/default.yaml"
datasets="socialiqa-train-dev" #"alphanli" #"physicaliqa-train-dev" # "socialiqa-train-dev"

for dataset in $datasets 
do
	python -m mowgli --dataset $dataset --output output/ --config $cfg 
done
