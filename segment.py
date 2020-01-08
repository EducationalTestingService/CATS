import argparse

import cats_predict
import cats_preprocess
import utils


def run_segmentation(input_dir, output_dir, embeddings, vocabulary):

    cats_preprocess.preprocess(input_dir, output_dir, vocabulary, ssplit=False)
    cats_predict.predict(output_dir, output_dir, embeddings, vocabulary, scores=False)

def main(input_dir, output_dir):
    embeddings, vocabulary = utils.load_models()
    run_segmentation(input_dir, output_dir, embeddings, vocabulary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
