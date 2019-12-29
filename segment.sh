#!/bin/bash

usage() { echo "Usage: $0 [-s 0|1] [-p 0|1] input_dir output_dir";}

SSPLIT=0
SCORES=0

while getopts ":s:p:" o; do
    case "${o}" in
        s)
            SSPLIT=${OPTARG}
            ;;
        p)
            SCORES=${OPTARG}
            ;;
        *)
            usage
            exit  
            ;;    
    esac
done
shift $((OPTIND-1))

IN_DIR=$1
OUT_DIR=$2

echo $IN_DIR
echo $OUT_DIR
echo $SSPLIT
echo $SCORES

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR

echo "Preprocessing the input..."
python $DIR/cats_preprocess.py $IN_DIR $OUT_DIR --ssplit $SSPLIT
echo "Preprocessing done, TFRECORDS and BLOCKS generated in the output directory."

echo "Making segmentation predictions and generating segmented texts..."
python $DIR/cats_predict.py $OUT_DIR $OUT_DIR --scores $SCORES
echo "Segmentation done, segmented texts stored in the output directory."

echo "Removing the intermediate serialized files..."
rm $OUT_DIR/records.tf
rm $OUT_DIR/blocks.pkl
echo "Completed. All done here, ciao bella!"








