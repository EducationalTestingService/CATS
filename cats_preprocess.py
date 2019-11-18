## This script preprocesses texts, creates and stores (seiralizes) instances (Tensorflow's TF Records) for classification (training or prediction)

import argparse
import os
import pickle

import config
import serializer
import utils


def preprocess(input_dir, output_dir, vocabulary, train=False, ssplit=False):

    if train is True and ssplit is False:
        print("For preparing training instances (that is, if --train 1), the text files need to be in the one-sentence-per-line format (must be set --ssplit 1)")
        exit()

    test_mode = (train is False)

    res = serializer.create_instances(input_dir,
                                      vocabulary,
                                      os.path.join(output_dir, "records.tf"),
                                      test=test_mode,
                                      title_start=config.seg_start,
                                      ssplit=(ssplit is True))
    pickle.dump(res, open(os.path.join(output_dir, "blocks.pkl"), "wb+"))
    print("All done here, ciao bella!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing texts and creating tfrecords (instances) for segmentation training or evaluation.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing text documents to be segmented")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the serialized tfrecords will be saved.")
    parser.add_argument("--train", type=int, default=0, help="Indicates if you're preparing tfrecords for training (value 1) the model or instances on which to predict (value 0) the segmentation scores.")
    parser.add_argument("--ssplit", type=int, default=1, help="Indicates whether the texts are already sentence split, one sentence per line in the text files (value 1), or the texts need to be first split for sentences (value 0)")
    args = parser.parse_args()

    _, vocabulary = utils.load_models()
    preprocess(args.input_dir, args.output_dir, vocabulary, train=True if args.train == 1 else False, ssplit=True if args.ssplit == 1 else False)
