###
# DATA
###

texts_lang = "en"
seg_start = "==========="
fake_sent = "fake sent 123: bla one bla day bla whatever."

# pre-trained word embs
vecs_dim = 300

vocab_path_en = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/embeddings/en.vocab"
vecs_path_en = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/embeddings/en.vectors"

vocab_path_lang = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/embeddings/hr.vocab"
vecs_path_lang = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/embeddings/hr.vectors"

###
# MODEL
###

MODEL_TYPE = "cats" # 'cats' or 'tlt'
MODEL_HOME = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/models/cats_pretrained" # for TLT, use "data/models/tlt_pretrained"

###
# ARCHITECTURE AND TRAINING
###

# general
batch_size = 20
sent_window = 16
sent_stride = 8
perc_blocks_train = 0.35
max_sent_len = 50
positional_embs_size = 10

# transformers
TOK_TRANS_PARAMS = {"num_hidden_layers" : 6,
                    "hidden_size" : vecs_dim + 2*positional_embs_size,
                    "num_heads" : 4, "filter_size" : 1024,
                    "relu_dropout" : 0.1,
                    "attention_dropout" : 0.1,
                    "layer_postprocess_dropout" : 0.1,
                    "allow_ffn_pad" : True
                    }

SENT_TRANS_PARAMS = {"num_hidden_layers" : 6,
                     "hidden_size" : vecs_dim + 2*positional_embs_size,
                     "num_heads" : 4,
                     "filter_size" : 1024,
                     "relu_dropout" : 0.1,
                     "attention_dropout" : 0.1,
                     "layer_postprocess_dropout" : 0.1,
                     "allow_ffn_pad" : True
                    }

TOK_TRANS_PARAMS_PREDICT = {"num_hidden_layers" : 6,
                    "hidden_size" : vecs_dim + 2*positional_embs_size,
                    "num_heads" : 4,
                    "filter_size" : 1024,
                    "relu_dropout" : 0,
                    "attention_dropout" : 0,
                    "layer_postprocess_dropout" : 0,
                    "allow_ffn_pad" : True
                    }

SENT_TRANS_PARAMS_PREDICT_CATS = {"num_hidden_layers" : 4,
                     "hidden_size" : vecs_dim + 2*positional_embs_size,
                     "num_heads" : 2,
                     "filter_size" : 1024,
                     "relu_dropout" : 0,
                     "attention_dropout" : 0,
                     "layer_postprocess_dropout" : 0,
                     "allow_ffn_pad" : True
                    }

SENT_TRANS_PARAMS_PREDICT_TLT = {"num_hidden_layers" : 6,
                     "hidden_size" : vecs_dim + 2*positional_embs_size,
                     "num_heads" : 4,
                     "filter_size" : 1024,
                     "relu_dropout" : 0,
                     "attention_dropout" : 0,
                     "layer_postprocess_dropout" : 0,
                     "allow_ffn_pad" : True
                    }

# training
tfrec_train = ""
EPOCHS = 100
SAVE_CHECKPOINT_STEPS = 500
