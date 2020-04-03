#! /bin/bash

mkdir -p output

#rm output/*
#rm checkpoint/*.mdl

cfg="cfg/trian.yaml"

python -m mowgli --config $cfg 
