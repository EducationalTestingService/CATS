import argparse

import cats_predict
import cats_preprocess


def run_segmentation(input_dir, output_dir):

    cats_preprocess.main(input_dir, output_dir, ssplit=False)
    cats_predict.main(output_dir, output_dir, scores=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()
    run_segmentation(args.input_dir, args.output_dir)
