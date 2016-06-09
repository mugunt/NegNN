# -*-coding:utf-8-*-
#! /usr/bin/env python

from bilstm import BiLSTM
# from random import shuffle
from NegNN.utils.tools import shuffle, padding
from NegNN.utils.metrics import *
from NegNN.processors import int_processor
from NegNN.processors import ext_processor
import tensorflow as tf
import numpy as np
import sys
import os
import time
import codecs


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("max_sent_length", 100, "Maximum sentence length for padding (default:100)")
tf.flags.DEFINE_integer("num_hidden", 200, "Number of hidden units per layer (default:200)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of y classes (default:2)")
# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate(default: 1e-4)")
tf.flags.DEFINE_boolean("scope_detection", True, "True if the task is scope detection or joined scope/event detection")
tf.flags.DEFINE_boolean("event_detection", False, "True is the task is event detection or joined scope/event detection")
tf.flags.DEFINE_integer("POS_emb",0,"0: no POS embeddings; 1: normal POS; 2: universal POS")
tf.flags.DEFINE_boolean("emb_update",False,"True if input embeddings should be updated (default: False)")
tf.flags.DEFINE_boolean("normalize_emb",False,"True to apply L2 regularization on input embeddings (default: False)")
# Data Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_boolean("pre_training", False, "True to use pretrained embeddings")
tf.flags.DEFINE_string("training_lang",'en', "Language of the tranining data (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def store_config(_dir,flags):
    with codecs.open(os.path.join(_dir,'config.ini'),'wb','utf8') as _config:
        for attr, value in sorted(FLAGS.__flags.items()):
            _config.write("{}={}\n".format(attr.upper(), value))

# Timestamp and output dir for current model
fold_name = "%s%s_%d%s" % ('PRE' if FLAGS.pre_training else "noPRE",
'upd' if FLAGS.pre_training and FLAGS.emb_update else '',
FLAGS.POS_emb,str(int(time.time())))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "NegNN","runs", fold_name))
print "Writing to {}\n".format(out_dir)

# Set checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
store_config(checkpoint_dir,FLAGS)

# Data Preparation
# ==================================================

# Load data
# if not FLAGS.pre_training:
#     train_set, dev_set, voc, voc_inv = int_processor.load_train_dev(FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang, out_dir)
# else:
#     train_set, dev_set, voc_inv, pre_emb_w, pre_emb_t = ext_processor.load_train_dev(FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang, FLAGS.embedding_dim, FLAGS.POS_emb)

if not FLAGS.pre_training:
    train_set, valid_set, voc, dic_inv = int_processor.load_train_dev(FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang, out_dir)
    vocsize = len(voc['w2idxs'])
    tag_voc_size = len(voc['t2idxs']) if POS_emb == 1 else len(voc['tuni2idxs'])
else:
    train_set, valid_set, dic_inv, pre_emb_w, pre_emb_t = ext_processor.load_train_dev(scope_dect, event_dect, tr_lang, emb_size, POS_emb)
    vocsize = pre_emb_w.shape[0]
    tag_voc_size = pre_emb_t.shape[0]

train_lex, train_tags, train_tags_uni, train_cue, _, train_y = train_set
valid_lex, valid_tags, valid_tags_uni, valid_cue, _, valid_y = valid_set

# Training
# ==================================================

# Decompose train and dev set
# train_lex, train_tags, train_tags_uni, train_cue, train_scope, train_y = train_set
# valid_lex, valid_tags, valid_tags_uni, valid_cue, valid_scope, valid_y = dev_set

# Set vocsize for words and tags
# if FLAGS.pre_training: voc_size = pre_emb_w.shape[0]
# else: voc_size = len(voc['w2idxs'])
# if FLAGS.POS_emb in [1,2]:
#     if FLAGS.pre_training:
#         tag_voc_size = pre_emb_t.shape[0]
#     else: tag_voc_size = len(voc['t2idxs']) if FLAGS.POS_emb == 1 else len(voc['tuni2idxs'])
# else: tag_voc_size = 0

def feeder(lex, cue, tags, _y, train = True):
    X = padding(lex, max_sent_len, vocsize - 1)
    C = padding(cue, max_sent_len, 2)
    if tags: T = padding(tags, max_sent_len, tag_voc_size - 1)
    Y = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],_y)).astype('int32'),max_sent_len,0,False)
    _mask = [1 if t!=vocsize - 1 else 0 for t in X]
    feed_dict={
        x: X,
        c: C,
        y: Y,
        istate_fw: numpy.zeros((1, 2*n_hidden)),
        istate_bw: numpy.zeros((1, 2*n_hidden)),
        seq_len: numpy.asarray([len(lex)]),
        mask: _mask}
    if tags: feed_dict.update{t:T}
    # if train:
    feed_dict.update({lr:clr})
    _, acc_train = sess.run([optimizer, accuracy], feed_dict = feed_dict)
    return acc_train
    # else:
    #     acc_test, pred = sess.run([accuracy,predictions], feed_dict = feed_dict)
    #     return acc_test, pred , Y

clr = FLAGS.learning_rate

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        bi_lstm = BiLSTM(num_hidden=FLAGS.num_hidden,
                num_classes=FLAGS.num_classes,
                voc_dim=voc_size,
                emb_dim=FLAGS.embedding_dim,
                sent_max_len = FLAGS.max_sent_length,
                tag_voc_dim = tag_voc_size,
                tags = True if FLAGS.POS_emb in [1,2] else False,
                external = FLAGS.pre_training,
                update = FLAGS.emb_update)

        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(clr).minimize(bi_lstm.loss)

        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        if pre_training:
            sess.run(_weights['w_emb'].assign(pre_emb_w))
            if FLAGS.POS_emb in [1,2]:
                sess.run(_weights['t_emb'].assign(pre_emb_t))

        best_f1 = 0.0
        for e in xrange(nepochs):

            # shuffle
            if FLAGS.POS_emb in [1,2]: shuffle([train_lex, train_tags, train_tags_uni, train_cue, train_y], 20)
            else: shuffle([train_lex,train_cue,train_y], 20)

            # TRAINING STEP
            train_tot_acc = []
            dev_tot_acc = []
            tic = time.time()
            for i in xrange(len(train_lex)):
                if FLAGS.POS_emb in [1,2]:
                    acc_train = feeder(train_lex[i],train_cue[i],train_tags[i] if POS_emb == 1 else train_tags_uni[i], train_y[i])
                else:
                    acc_train = feeder(train_lex[i], train_cue[i], None, train_y[i])
                # Calculating batch accuracy
                train_tot_acc.append(acc_train)
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()               
            print "TRAINING MEAN ACCURACY: ", sum(train_tot_acc)/len(train_lex)

            # DEVELOPMENT STEP
            pred_dev = []
            gold_dev = []
            for i in xrange(len(valid_lex)):
                if FLAGS.POS_emb in [1,2]:
                    acc_dev, pred, Y_dev = feeder(valid_lex[i],valid_cue[i],valid_tags[i] if POS_emb == 1 else valid_tags_uni[i],valid_y[i],train=False)
                else:
                    acc_dev, pred, Y_dev = feeder(valid_lex[i],valid_cue[i],None, valid_y[i],train=False)
                dev_tot_acc.append(acc_dev)
                pred_dev.append(pred[:len(valid_lex[i])])
                gold_dev.append(Y_dev[:len(valid_lex[i])])
            print 'DEV MEAN ACCURACY: ',sum(dev_tot_acc)/len(valid_lex)
            f1,rep_dev,cm_dev = get_eval(pred_dev,gold_dev)

            # STORE INTERMEDIATE RESULTS
            if f1 > best_f1:
                best_f1 = f1
                print "Best f1 is: ",best_f1
                be = e
                # store the model
                checkpoint_prefix = os.path.join(folder, "model")
                saver.save(sess, checkpoint_prefix,global_step=be)
                print "Model saved."
                write_report(folder,rep_dev,cm_dev,'dev')
                store_prediction(folder, valid_lex, dic_inv, pred_dev, gold_dev, 'dev')
                dry = 0
            else:
                dry += 1

            if abs(be-e) >= 10 and dry>=5:
                print "Halving the lr..."
                clr *= 0.5
                dry = 0



        # # Set the external matrices, if external flag is True
        # if FLAGS.pre_training:
        #     sess.run(bi_lstm._weights['w_emb'].assign(pre_emb_w))
        #     if FLAGS.POS_emb in [1,2]:
        #         sess.run(bi_lstm._weights['t_emb'].assign(pre_emb_t))
        # try:
        #     best_f1 = 0.0
        #     be = 0

        #     for e in xrange(FLAGS.num_epochs):
        #         # shuffle
        #         shuffle([train_lex, train_tags, train_tags_uni, train_cue, train_scope, train_y], 20)
        #         tic = time.time()
        #         train_tot_acc = []
        #         dev_tot_acc = []
        #         for i in xrange(len(train_lex)):
        #             X = padding(train_lex[i],FLAGS.max_sent_length,voc_size - 1)
        #             C = padding(train_cue[i],FLAGS.max_sent_length,2)
        #             if FLAGS.POS_emb == 1:
        #                 T = padding(train_tags[i],FLAGS.max_sent_length,tag_voc_size - 1)
        #             if FLAGS.POS_emb == 2:
        #                 T = padding(train_tags_uni[i],FLAGS.max_sent_length,tag_voc_size - 1)
        #             Y = padding(np.asarray(map(lambda x: [1,0] if x == 0 else [0,1],train_y[i])).astype('int32'),FLAGS.max_sent_length,0,False)
        #             _mask = [1 if _t!=voc_size else 0 for _t in X]
        #             # build feed_dict
        #             feed_dict = {
        #                 bi_lstm.x: X,
        #                 bi_lstm.c: C,
        #                 bi_lstm.y: Y,
        #                 bi_lstm.istate_fw: np.zeros((1, 2*FLAGS.num_hidden)),
        #                 bi_lstm.istate_bw: np.zeros((1, 2*FLAGS.num_hidden)),
        #                 bi_lstm.seq_len: np.asarray([len(train_lex[i])]),
        #                 bi_lstm.mask: _mask,
        #                 bi_lstm.lr: clr
        #                 }
        #             if FLAGS.POS_emb in [1,2]:
        #                 feed_dict.update({bi_lstm.t:T})
        #             # run training
        #             sess.run(optimizer,feed_dict = feed_dict)
        #             accuracy = sess.run(bi_lstm.accuracy,feed_dict = feed_dict)
        #             # normalization of embeddings if needed
        #             # if FLAGS.normalize_emb:
        #             #     sess.run(bi_lstm.normalize_w_emb)
        #                 # if FLAGS.POS_emb in [1,2]: 
        #                 #     sess.run(bi_lstm.normalize_t_emb)
        #             train_tot_acc.append(get_accuracy(accuracy,len(train_lex[i])))
        #             print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
        #             sys.stdout.flush()
        #         print "Mean training accuracy is: ",sum(train_tot_acc)/len(train_tot_acc)

        #         pred_dev, gold_dev = [],[]
        #         for i in xrange(len(valid_lex)):
        #             # pad the word and cue vector of the dev set with random shit
        #             X_dev = padding(valid_lex[i],FLAGS.max_sent_length,voc_size - 1)
        #             C_dev = padding(valid_cue[i],FLAGS.max_sent_length,2)
        #             if FLAGS.POS_emb == 1:
        #                 T_dev = padding(valid_tags[i],FLAGS.max_sent_length,tag_voc_size - 1)
        #             if FLAGS.POS_emb == 2:
        #                 T_dev = padding(valid_tags_uni[i],FLAGS.max_sent_length,tag_voc_size - 1)
        #             Y_dev = padding(np.asarray(map(lambda x: [1,0] if x == 0 else [0,1],valid_y[i])).astype('int32'),FLAGS.max_sent_length,0,False)
        #             _mask_dev = [1 if t_d!=voc_size else 0 for t_d in X_dev]
        #             feed_dict={
        #                 bi_lstm.x: X_dev,
        #                 bi_lstm.c: C_dev,
        #                 bi_lstm.y: Y_dev,
        #                 bi_lstm.istate_fw: np.zeros((1, 2*FLAGS.num_hidden)),
        #                 bi_lstm.istate_bw: np.zeros((1, 2*FLAGS.num_hidden)),
        #                 bi_lstm.seq_len: np.asarray([len(valid_lex[i])]),
        #                 bi_lstm.mask: _mask_dev
        #                 }
        #             if FLAGS.POS_emb in [1,2]:
        #                 feed_dict.update({bi_lstm.t: T_dev})
        #             # get dev set accuracy
        #             accuracy_dev,label_dev = sess.run([bi_lstm.accuracy,bi_lstm.label_out], feed_dict = feed_dict )
        #             dev_tot_acc.append(get_accuracy(accuracy_dev,len(valid_lex[i])))

        #             pred_dev.append(label_dev[:len(valid_lex[i])])
        #             gold_dev.append(Y_dev[:len(valid_lex[i])])
        #         print 'Mean dev accuracy is: ',sum(dev_tot_acc)/len(valid_lex)
        #         print pred_dev[1]
        #         print gold_dev[1]
        #         print pred_dev[3]
        #         print gold_dev[3]
        #         print pred_dev[12]
        #         print gold_dev[12]
        #         print pred_dev[23]
        #         print gold_dev[23] 
        #         f1,rep_dev,cm_dev = get_eval(pred_dev,gold_dev)

        #         if f1 > best_f1:
        #             best_f1 = f1
        #             print "Best f1 is: ",best_f1
        #             be = e

        #             # store the model
        #             saver.save(sess, checkpoint_prefix, global_step=be)
        #             print "Model saved."

        #             print "Storing reports..."
        #             with codecs.open(os.path.join(checkpoint_dir,'valid_report.txt'),'wb','utf8') as store_rep_dev:
        #                 store_rep_dev.write(rep_dev)
        #                 store_rep_dev.write(str(cm_dev)+"\n")
        #             print "Reports stored..."

        #             print "Storing labelling results for dev set..."
        #             with codecs.open(os.path.join(checkpoint_dir,'best_valid.txt'),'wb','utf8') as store_pred:
        #                 for s, y_sys, y_hat in zip(valid_lex,pred_dev,gold_dev):
        #                     s = [voc_inv['idxs2w'][w] if w in voc_inv['idxs2w'] else '<UNK>' for w in s]
        #                     assert len(s)==len(y_sys)==len(y_hat)
        #                     for _word,_sys,gold in zip(s,y_sys,y_hat):
        #                         _p = list(_sys).index(_sys.max())
        #                         _g = 0 if list(gold)==[1,0] else 1
        #                         store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
        #                     store_pred.write("\n")
        #             dry = 0
        #         else:
        #             dry += 1

        #         if abs(be-e) >= 10 and dry>=5:
        #             print "Halving the lr..."
        #             clr *= 0.5
        #             dry = 0
        # except KeyboardInterrupt:
        #     print "Exiting safely(?)..."
        #     sys.exit()
