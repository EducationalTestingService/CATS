## Script that trains an instance of a CATS (or TLT) model

import tensorflow as tf
import model
import serializer
import config
import utils
import numpy as np
import get_data
import pickle
import os

dirname = dirname = os.path.dirname(os.path.realpath(__file__))
print("Dirname: " + dirname)

print("Loading word embeddings...")
embs = utils.load_vectors(os.path.join(dirname, config.vecs_path_en))
vocab = utils.load_vocab(os.path.join(dirname, config.vocab_path_en))
print("Loaded.")

print("Defining estimator...")
rconf = tf.estimator.RunConfig(save_checkpoints_steps=config.SAVE_CHECKPOINT_STEPS, 
                               save_checkpoints_secs=None, 
                               model_dir=os.path.join(dirname, config.MODEL_HOME))
print("Defined.")

params = {"padding_value" : vocab["<PAD>"], "wembs" : embs, "vocab" : vocab, "coherence_hinge_margin" : 1, "learning_rate" : 0.0001}
estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=rconf, params=params)

print("Training the model...")
res = estimator.train(input_fn=lambda : get_data.get_data(os.path.join(dirname, config.tfrec_train), is_train = True, epochs = config.EPOCHS))