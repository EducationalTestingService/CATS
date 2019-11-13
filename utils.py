import codecs
import os
import pickle

import numpy as np

import config


def load_txt_file(path):
  return (codecs.open(path, 'r', encoding = 'utf8', errors = 'replace')).read()

def load_lines(path):
  return [l.strip() for l in list(codecs.open(path, "r", encoding = 'utf8', errors = 'replace').readlines())]

def load_vocab(path):
  return pickle.load(open(path,"rb"))

def load_vectors(path):
  return np.load(path)


def get_files_recursive(path, files = []):
  print(path + "\t" + str(len(files)))
  ## tmp, for testing
  #if len(files) > 5000:
  #  return
  items = os.listdir(path)
  for i in items:
    if os.path.isfile(os.path.join(path, i)):
      files.append(os.path.join(path, i))
    elif os.path.isdir(os.path.join(path, i)):
      get_files_recursive(os.path.join(path, i), files)

def write_list(path, list):
  f = codecs.open(path,'w',encoding='utf8')
  for l in list:
    f.write(l + "\n")
  f.close()



def load_models():
    embs = load_vectors(config.vecs_path_en)
    vocab = load_vocab(config.vocab_path_en)
    print("Loaded")

    if config.texts_lang != "en":
        print("Loading " + config.texts_lang.upper() + " embeddings...")
        embs_tgt = load_vectors(config.vecs_path_lang)
        vocab_tgt = load_vectors(config.vocab_path_lang)

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
    return embs_tgt, vocab_tgt
