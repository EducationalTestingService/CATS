import tensorflow as tf
import numpy as np
from transformer.model import transformer
from transformer.model import model_utils
import config

def model_fn(features, labels, mode, params):
  """Defines the hierarchical transformer model for segmentation and coherence"""

  print("Training the model: " + config.MODEL_TYPE.upper())
  print("Defining the model...")
  with tf.variable_scope("model"):
    embeddings = params["wembs"].astype(np.float32)
    vocab = params["vocab"]

    print("Creating positional embeddings...")    
    # Creating positional embeddings (parameters), both for sentence-level positions and paragraph-level positions
    par_pos_embs = tf.get_variable("par_pos_embs", shape=[config.max_sent_len * config.sent_window, config.positional_embs_size], 
                                        initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    sent_pos_embs = tf.get_variable("sent_pos_embs", shape=[config.max_sent_len, config.positional_embs_size], 
                                        initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

    # Looking up word embeddings and positional embeddings
    input_true = tf.reshape(features["true_seqs"], [config.batch_size * config.sent_window, config.max_sent_len])
    input_fake = tf.reshape(features["fake_seqs"], [config.batch_size * config.sent_window, config.max_sent_len])
    true_seq = tf.nn.embedding_lookup(embeddings, input_true)
    fake_seq = tf.nn.embedding_lookup(embeddings, input_fake)
 
    b_sent_pos_embs = tf.nn.embedding_lookup(sent_pos_embs, np.tile(list(range(config.max_sent_len)), (config.batch_size * config.sent_window, 1)))
    b_par_pos_embs = tf.nn.embedding_lookup(par_pos_embs, np.tile(np.reshape(list(range(config.max_sent_len * config.sent_window)), (config.sent_window, config.max_sent_len)), (config.batch_size, 1)))
    
    inputs_true_seq = tf.concat([true_seq, b_sent_pos_embs, b_par_pos_embs], axis = 2)
    inputs_fake_seq = tf.concat([fake_seq, b_sent_pos_embs, b_par_pos_embs], axis = 2)    

    print("Token-level transformer...")    
    # Encoder stack for tokens
    with tf.variable_scope("tok_trans"):
      enc_stack_toks = transformer.EncoderStack(config.TOK_TRANS_PARAMS if mode == tf.estimator.ModeKeys.TRAIN else config.TOK_TRANS_PARAMS_PREDICT, mode)

      attention_bias_true = model_utils.get_padding_bias(input_true, padding_value = params["padding_value"])
      inputs_padding_true = model_utils.get_padding(input_true, padding_value = params["padding_value"])
      attention_bias_fake = model_utils.get_padding_bias(input_fake, padding_value = params["padding_value"])
      inputs_padding_fake = model_utils.get_padding(input_fake, padding_value = params["padding_value"])
    
      transformed_toks_true = enc_stack_toks(inputs_true_seq, attention_bias_true, inputs_padding_true) 
      transformed_toks_fake = enc_stack_toks(inputs_fake_seq, attention_bias_fake, inputs_padding_fake)
    
    # taking transformer representations of the first and last token as sentence reps
    sents_true = transformed_toks_true[:, 0, :]
    sents_true = tf.reshape(sents_true, [config.batch_size, config.sent_window, -1])
   
    sents_fake = transformed_toks_fake[:, 0, :]
    sents_fake = tf.reshape(sents_fake, [config.batch_size, config.sent_window, -1])   

    # adding fake "token" for the start of sequence of sentences (for the second transformer)
    sst = np.concatenate((embeddings[vocab["<SS>"]], embeddings[vocab["<SSS>"]][: 2 * config.positional_embs_size]))
    sent_seq_start = tf.reshape(tf.tile(sst, [config.batch_size]), [config.batch_size, 1, -1])
    seq_correct = tf.concat([sent_seq_start, sents_true], axis = 1)
    seq_fake = tf.concat([sent_seq_start, sents_fake], axis = 1)

    print("Sentence-level transformer...")    
    # Encoder stack for sentences
    with tf.variable_scope("sent_trans"):
      enc_stack_sents = transformer.EncoderStack(config.SENT_TRANS_PARAMS if mode == tf.estimator.ModeKeys.TRAIN else (config.SENT_TRANS_PARAMS_PREDICT_CATS if config.MODEL_TYPE == "cats" else config.SENT_TRANS_PARAMS_PREDICT_TLT), mode)
      sent_att_bias = model_utils.get_padding_bias(tf.zeros([config.batch_size, config.sent_window + 1], tf.float32), padding_value = -1)
      sent_inputs_padd = model_utils.get_padding(tf.zeros([config.batch_size, config.sent_window + 1], tf.float32), padding_value = -1)
      
      transformed_sents_true = enc_stack_sents(seq_correct, sent_att_bias, sent_inputs_padd) 
      transformed_sents_fake = enc_stack_sents(seq_fake, sent_att_bias, sent_inputs_padd) 

    rep_true = transformed_sents_true[:, 0, :]
    rep_fake = transformed_sents_fake[:, 0, :]

    sent_reps_true = transformed_sents_true[:, 1:, :]

    # segmentation classification
    print("Segmentation classifier...")    
    with tf.variable_scope("seg_classifier"):
      w_seg = tf.get_variable("w_seg", shape=[config.vecs_dim + 2*config.positional_embs_size, 2], 
                                       initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
      b_seg = tf.get_variable("b_seg", initializer=tf.zeros(2), dtype = tf.float32)

    seg_probs = tf.nn.softmax(tf.add(tf.tensordot(sent_reps_true, w_seg, axes = [[2], [0]]), b_seg))
    seg_labs = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1), lambda: tf.constant([0, 1], dtype=tf.float32),
                                  lambda: tf.constant([1, 0], dtype=tf.float32)), tf.cast(features["seg_lab"], tf.float32))
    seg_labs = tf.reshape(seg_labs, [config.batch_size, config.sent_window, -1])

    # coherence contrast
    print("Coherence regressor...")    
    with tf.variable_scope("coherence_regressor"): 
      w_coh = tf.get_variable("w_coh", shape=[config.vecs_dim + 2*config.positional_embs_size], 
                                       initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
      b_coh = tf.get_variable("b_coh", initializer=tf.zeros(1), dtype = tf.float32)
      scores_true = tf.reshape(tf.add(tf.tensordot(rep_true, w_coh, [[1], [0]]), b_coh), [-1, 1])
      scores_fake = tf.reshape(tf.add(tf.tensordot(rep_fake, w_coh, [[1], [0]]), b_coh), [-1, 1])
      norm_scores = tf.nn.softmax(tf.concat([scores_true, scores_fake], axis = 1))
      norm_scores_true = norm_scores[:, 0]
      norm_scores_fake = norm_scores[:, 1]
      scores_diff = tf.subtract(norm_scores_true, norm_scores_fake)
      
    # losses
    print("Defining losses...")    
    loss_seg = -1 * tf.reduce_sum(tf.multiply(tf.log(seg_probs), seg_labs))
    if config.MODEL_TYPE == "cats":
      loss_coh = tf.reduce_sum(tf.maximum(tf.constant(0, dtype = tf.float32), tf.constant(params["coherence_hinge_margin"], dtype = tf.float32) - scores_diff))
      loss_sum = loss_seg + loss_coh

    tf.summary.scalar('seg_loss', loss_seg)
    if config.MODEL_TYPE == "cats":
      tf.summary.scalar('coh_loss', loss_coh)

    # train_operations
    print("Defining train operations...")     
    if mode == tf.estimator.ModeKeys.TRAIN:
      print("Defining segmentation optimization...")  
      optimizer_seg = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
      train_op_seg = optimizer_seg.minimize(loss_seg, global_step=tf.train.get_global_step())

      if config.MODEL_TYPE == "cats":
        print("Defining coherence optimization...")
        optimizer_coh = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op_coh = optimizer_coh.minimize(loss_coh, global_step=tf.train.get_global_step())
        train_op = tf.group(train_op_seg, train_op_coh)
        return tf.estimator.EstimatorSpec(mode, loss=loss_sum, train_op=train_op, eval_metric_ops={})
      else:
        return tf.estimator.EstimatorSpec(mode, loss=loss_seg, train_op=train_op_seg, eval_metric_ops={})
 
    else:
      print("Evaluating (predicting)...")
      preds = tf.argmax(seg_probs, axis = 0)
      return tf.estimator.EstimatorSpec(mode, loss=loss_seg, predictions = seg_probs, eval_metric_ops={})

    