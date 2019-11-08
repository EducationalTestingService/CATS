import argparse

import cats_predict
import cats_preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()
    cats_preprocess.main(args.input_dir, args.output_dir, ssplit=False)
    cats_predict.main(args.output_dir, args.output_dir, scores=True)
