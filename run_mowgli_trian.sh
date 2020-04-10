#! /bin/bash

mkdir -p output

#rm output/*
#rm checkpoint/*.mdl

cfg=$1

python -m mowgli --config $cfg 
