## This script preprocesses texts, creates and stores (seiralizes) instances (Tensorflow's TF Records) for classification (training or prediction)

import serializer
import config
import utils
import numpy as np
import get_data
import pickle
from sys import stdin
import os
import argparse
import model
import tensorflow as tf

parser = argparse.ArgumentParser(description="Preprocessing texts and creating tfrecords (instances) for segmentation training or evaluation.")
parser.add_argument("input_dir", type=str, help="Path to the directory containing text documents to be segmented")
parser.add_argument("output_dir", type=str, help="Path to the directory where the serialized tfrecords will be saved.")
parser.add_argument("--train", type=int, default=0, help="Indicates if you're preparing tfrecords for training (value 1) the model or instances on which to predict (value 0) the segmentation scores.")
parser.add_argument("--ssplit", type=int, default=1, help="Indicates whether the texts are already sentence split, one sentence per line in the text files (value 1), or the texts need to be first split for sentences (value 0)")
args = parser.parse_args()

if args.train == 1 and args.ssplit == 0:
  print("For preparing training instances (that is, if --train 1), the text files need to be in the one-sentence-per-line format (must be set --ssplit 1)")
  exit()

dirname = dirname = os.path.dirname(os.path.realpath(__file__))
lang = config.texts_lang

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

# Serialization is different based on test/train
test_mode = (args.train == 0)

res = serializer.create_instances(args.input_dir, vocab_tgt, os.path.join(args.output_dir, "records.tf"), test = test_mode, title_start = config.seg_start, ssplit = (args.ssplit == 1))
pickle.dump(res, open(os.path.join(args.output_dir, "blocks.pkl"),"wb+"))
print("All done here, ciao bella!")
