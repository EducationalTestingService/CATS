#!/bin/bash

#$ -S /bin/bash
#$ -j y
#$ -q gpu.q

conda activate textsegenv
cd /home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/cats_deploy

./segment.sh -s 0 -p 1 $1 $2
#CUDA_VISIBLE_DEVICES=0 python cats_predict.py data/datasets/en/tfrec/ data/datasets/en/segmented/wiki50-segmented-ets/
