#!/bin/bash

#Find out where we are
BASE=`pwd`

for dkmax in 1 2; do 
    python tempo.py -i params_tempo_dkmax=${dkmax}.txt -o output_tempo_dkmax=${dkmax}.txt
done

