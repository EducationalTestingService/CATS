import argparse
import os
import pickle

import numpy as np

import config
import get_data
import model
import tensorflow as tf
import utils


def predict(input_dir, output_dir, embeddings, vocabulary, scores=False):
    dirname = os.path.dirname(os.path.realpath(__file__))
    print(dirname)

    blocks_path = os.path.join(input_dir, "blocks.pkl")
    records_path = os.path.join(input_dir, "records.tf")
    write_pred_score = (scores is True)

    print("Defining estimator...")
    rconf = tf.estimator.RunConfig(save_checkpoints_steps=config.SAVE_CHECKPOINT_STEPS,
                                   save_checkpoints_secs=None,
                                   model_dir=os.path.join(dirname, config.MODEL_HOME))

    params = {"padding_value": vocabulary["<PAD>"],
              "wembs": embeddings,
              "vocab": vocabulary,
              "coherence_hinge_margin" : 1,
              "learning_rate" : 0.0001}
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=rconf, params=params)

    print("Loading serialized raw test set texts...")
    test_texts = pickle.load(open(blocks_path,"rb"))
    print("Loaded.")

    res = estimator.predict(input_fn=lambda : get_data.get_data(records_path, is_train = False, epochs = 1))

    print("Documents to segment: " + str(len(test_texts[0])))
    flat_blocks = []
    for x in test_texts[0]:
        print(len(x[1]))
        flat_blocks.extend(x[1])

    print("Number of prediction blocks: " + str(len(flat_blocks)))

    print("Predicting with the model (this may take a while, depending on the number of documents)...")
    res_list = list(res)
    print("Predictions completed.")

    thold = 0.3 if config.MODEL_TYPE == "cats" else 0.5

    glob_cntr = 0
    docs = test_texts[0]

    agg_docs = []

    for i in range(len(docs)):
        fname = docs[i][0]
        if i % 1000 == 1:
            print(fname)
            print(str(i) + " of " + str(len(docs)) + " documents...")
        blocks = docs[i][1]
        preds_blocks = res_list[glob_cntr: glob_cntr + len(blocks)]
        glob_cntr += len(blocks)

        sent_scores = [(b[0][0], b[0][1], []) for b in blocks]
        for b_ind in range(len(blocks)):
            for relb_ind in range(len(blocks[b_ind])):
                if blocks[b_ind][relb_ind][0] == config.fake_sent:
                    break
                else:
                    sent_ind = b_ind + relb_ind
                    score = preds_blocks[b_ind][relb_ind][1]
                    sent_scores[sent_ind][2].append(score)
        agg_sent_scores = [(x[0], x[1], np.mean(x[2]), (1 if np.mean(x[2]) >= thold else 0)) for x in sent_scores]
        agg_docs.append(agg_sent_scores)

    # printing out predictions
    docnames = [x[0] for x in docs]
    print("Storing segmented texts...")
    docscores = zip(docnames, agg_docs)
    for name, sentscores in docscores:
        print("Document: " + name)
        lines = []
        for s in sentscores:
            if s[2] >= thold:
                lines.append(config.seg_start)
            lines.append(s[0] + "\t" + str(s[2]) if write_pred_score else s[0])
        utils.write_list(os.path.join(output_dir, name.split("/")[-1] + ".seg"), lines)
    print("Stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmenting text documents from a given directory.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing text documents to be segmented")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the serialized tfrecords will be saved.")
    parser.add_argument("--scores", type=int, default=0, help="Indicates whether to print segmentation prediction probabilities next to each sentence (default 0 = false).")

    args = parser.parse_args()
    embeddings, vocabulary = utils.load_models()
    predict(args.input_dir, args.output_dir, embeddings, vocabulary, scores=False if args.scores == 0 else True)
