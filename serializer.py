import utils
import config
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
import random
import tensorflow as tf
from sys import stdin

def tokenize(string):
  return WordPunctTokenizer().tokenize(string)

def sentence_split(text):
  return sent_tokenize(text)  

def create_instances(dir_path, vocab, records_path, test = False, title_start = "========,", ssplit = True):
  writer = tf.python_io.TFRecordWriter(records_path)
  files = []
  utils.get_files_recursive(dir_path, files)
  print("Found " + str(len(files)) + " text files.")
  cnt = 0
  total = 0

  if test:
    doc_blocks = []

  for f in files:
    cnt += 1
    print("File: " + str(cnt))
    records, rb_num, windows = process_document(f, vocab, test = test, title_start = title_start, ssplit = ssplit)
    if test:
      doc_blocks.append((f, windows))
    total += rb_num
    for r in records:
      writer.write(r.SerializeToString())
    if cnt % 10 == 0:
      print("Current num. blocks: " + str(total))
  print("Total training blocks: " + str(total))

  if test: 
    fake_ids = to_token_ids([(config.fake_sent, 0)], vocab)[0]
    print(fake_ids)
    print(total)
    fake_recs = (config.batch_size - (total % config.batch_size)) * config.sent_window
    print(fake_recs)
    for i in range(fake_recs):
      fr = fake_record(fake_ids)
      writer.write(fr.SerializeToString())
    return doc_blocks, fake_recs
  
def process_document(path, vocab, title_start = "========,", forbidden_start = "***LIST***", test = False, ssplit = True):
  print("ssplit: " + str(ssplit))
  lines = ([l for l in utils.load_lines(path) if not l.startswith(forbidden_start)]) if ssplit else (sentence_split(utils.load_txt_file(path)))
  stride = 1 if test else config.sent_stride
  lab_lines = []
  lines_txt = []
  for i in range(len(lines)):
    if lines[i].startswith(title_start):
      continue 
    if (i-1) >= 0 and lines[i-1].startswith(title_start):
      lab_lines.append((lines[i], 1))
    else:
      lab_lines.append((lines[i], 0))  
    lines_txt.append(lines[i])

  raw_blocks = []
  i = 0
  while i < len(lab_lines):
    block = lab_lines[i : i + config.sent_window]
    if len(block) < config.sent_window:
      block.extend([(config.fake_sent, 0)] * (config.sent_window - len(block)))  
    raw_blocks.append(block)
    i += stride

  if not test: 
    random.shuffle(raw_blocks)
    raw_blocks = raw_blocks[:int(config.perc_blocks_train * len(raw_blocks))]

  doc_recs = []
  for rb in raw_blocks:
    records = create_one_instance(rb, lines_txt, vocab)  
    doc_recs.extend(records)
  
  return doc_recs, len(raw_blocks), raw_blocks if test else None

def create_fake_block(block, lines):
  block_fake = block.copy()
  random.shuffle(block_fake)
  p = random.random()
  if p >= 0.5:
    for i in range(len(block_fake)):
      p = random.random()
      if p >= 0.5:
        l = lines[random.randint(0, len(lines) - 1)]
        block_fake[i] = (l, 0)
  return block_fake 

def to_token_ids(block, vocab):
  sentences = []
  for i in range(len(block)):
    toks = [(t if t in vocab else (t.lower() if t.lower() in vocab else "<UNK>")) for t in tokenize(block[i][0])] + ["<S/>"]
    if len(toks) < config.max_sent_len:
      toks = toks + ["<PAD>"] * (config.max_sent_len - len(toks))
    elif len(toks) > config.max_sent_len:
      toks = toks[: config.max_sent_len]
    ids = [(vocab[t] if t in vocab else vocab[t.lower()]) for t in toks]  
    #print(ids)
    #stdin.readline()
    sentences.append(ids) 
  return sentences    
    
def create_one_instance(block, lines, vocab):
  records = []
  fake_block = create_fake_block(block, lines)
  block_toks = to_token_ids(block, vocab)
  seg_labs = [x[1] for x in block] 
  fake_block_toks = to_token_ids(fake_block, vocab)
  
  sent_insts = list(zip(block_toks, fake_block_toks, seg_labs))
  for item in sent_insts:
    r = tf.train.SequenceExample()
    r.context.feature["seg_lab"].int64_list.value.append(int(item[2]))

    true_seq = r.feature_lists.feature_list["true_seqs"]
    fake_seq = r.feature_lists.feature_list["fake_seqs"]

    for tid in item[0]:
      true_seq.feature.add().int64_list.value.append(tid)
    for tid in item[1]:
      fake_seq.feature.add().int64_list.value.append(tid)
    records.append(r)
  return records

def fake_record(fake_ids):
  r = tf.train.SequenceExample()
  r.context.feature["seg_lab"].int64_list.value.append(int(0))

  true_seq = r.feature_lists.feature_list["true_seqs"]
  fake_seq = r.feature_lists.feature_list["fake_seqs"]

  for i in range(len(fake_ids)):
    true_seq.feature.add().int64_list.value.append(fake_ids[i])
    fake_seq.feature.add().int64_list.value.append(fake_ids[i])
  return r
    
  
   
  