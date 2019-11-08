## This script preprocesses texts, creates and stores (seiralizes) instances (Tensorflow's TF Records) for classification (training or prediction)

import os
import pickle

import numpy as np

import config
import serializer
import utils


def main(input_dir, output_dir, train=False, ssplit=False):

    if train is True and ssplit is False:
        print("For preparing training instances (that is, if --train 1), the text files need to be in the one-sentence-per-line format (must be set --ssplit 1)")
        exit()

    dirname = dirname = os.path.dirname(os.path.realpath(__file__))

    print("Loading EN embeddings...")
    embs = utils.load_vectors(os.path.join(dirname, config.vecs_path_en))
    vocab = utils.load_vocab(os.path.join(dirname, config.vocab_path_en))
    print("Loaded")

    if config.texts_lang != "en":
        print("Loading " + config.texts_lang.upper() + " embeddings...")
        embs_tgt = utils.load_vectors(os.path.join(dirname, config.vecs_path_lang))
        vocab_tgt = utils.load_vectors(os.path.join(dirname, config.vocab_path_lang))
        print("Loaded. Padding with special tokens.")

        print(len(embs_tgt))
        print(len(vocab_tgt))

        special_tokens = ["<PAD>", "<UNK>", "<S>", "<S/>", "<SS>", "<SSS>"]
        for st in special_tokens:
            vocab_tgt[st] = len(vocab_tgt)
            embs_tgt = np.vstack((embs_tgt, [embs[vocab[st]]]))

        print(len(embs_tgt))
        print(len(vocab_tgt))
        print("Padded.")

    else:
        embs_tgt = embs
        vocab_tgt = vocab

    print("To create TFRECORD instances...")

    test_mode = (train is False)

    res = serializer.create_instances(input_dir,
                                      vocab_tgt,
                                      os.path.join(output_dir, "records.tf"),
                                      test=test_mode,
                                      title_start=config.seg_start,
                                      ssplit=(ssplit is True))
    pickle.dump(res, open(os.path.join(output_dir, "blocks.pkl"), "wb+"))
    print("All done here, ciao bella!")
