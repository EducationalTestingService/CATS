import argparse
import os
import tempfile
from os.path import join

import segment
import utils


def segment_text(input_text):
    with tempfile.TemporaryDirectory() as temp_dir:

        input_dir = join(temp_dir, 'input')
        output_dir = join(temp_dir, 'output')
        for dirname in [input_dir, output_dir]:
            os.makedirs(dirname)
        with open(join(input_dir, 'input.txt'), 'w') as wf:
            wf.write(input_text)

        embeddings, vocabulary = utils.load_models()
        segment.run_segmentation(input_dir, output_dir, embeddings, vocabulary)

        segmented_text = open(join(output_dir, 'input.txt.seg')).readlines()
        return segmented_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text")

    args = parser.parse_args()
    segmented_text = segment_text(args.input_text)
    print(segmented_text)
