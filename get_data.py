import tensorflow as tf
import config

def _get_dataset(paths, is_train, epochs, deserialize_fn):
  dataset = tf.data.TFRecordDataset(paths)
  if is_train:
    dataset = dataset.repeat(epochs)
    dataset = dataset.map(deserialize_fn, num_parallel_calls=32)
  else:
    dataset = dataset.repeat(1)
    dataset = dataset.map(deserialize_fn)
  return dataset

def get_data(path, is_train = True, epochs = 1):
  dataset = _get_dataset([path], is_train = is_train, epochs = epochs, deserialize_fn = deserialize_data)
  dataset = dataset.batch(config.batch_size * config.sent_window, drop_remainder=True)     
  dataset = dataset.prefetch(5)
  dataset = dataset.make_one_shot_iterator().get_next()
  
  features = {
   'seg_lab': dataset['seg_lab'],
   'true_seqs': dataset['true_seqs'],
   'fake_seqs': dataset['fake_seqs'],
  }

  return features, None
  
def deserialize_data(example_proto):
  context_features = {
    "seg_lab": tf.FixedLenFeature([], dtype=tf.int64),
  }

  sequence_features = {
    "true_seqs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "fake_seqs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  }

  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=example_proto,
    context_features=context_features,
    sequence_features=sequence_features
  )

  features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
  return features