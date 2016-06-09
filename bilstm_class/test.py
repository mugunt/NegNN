# -*-coding:utf-8-*-
#! /usr/bin/env python

from bilstm import BiLSTM
from random import shuffle
from NegNN.utils.tools import shuffle, padding
from NegNN.utils.metrics import *
from NegNN.processors import int_processor
from NegNN.processors import ext_processor

import tensorflow as tf
import numpy as np
import codecs
import sys
import os


# Parameters
# ==================================================
# Model Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_string("checkpoint_dir",'',"Path to the directory where the last checkpoint is stored")
tf.flags.DEFINE_string("test_name",'current',"Name to assign to report and test files")
tf.flags.DEFINE_string("test_lang",'en', "en for English, it for Italian (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def retrieve_flags(config_file):
    _config = codecs.open(config_file,'rb','utf8').readlines()
    return dict([(k.lower(),v) for k,v in map(lambda x: x.strip().split('='),_config)])

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
config_file = os.path.join(FLAGS.checkpoint_dir,'config.ini')
model_flags = retrieve_flags(config_file)

def str2bool(x):
    return True if x == "True" else False

scope_detection = str2bool(model_flags['scope_detection'])
event_detection = str2bool(model_flags['event_detection'])
embedding_dim = int(model_flags['embedding_dim'])
POS_emb = int(model_flags['pos_emb'])
pre_training = str2bool(model_flags['pre_training'])
num_classes = int(model_flags['num_classes'])
max_sent_length = int(model_flags['max_sent_length'])
num_hidden = int(model_flags['num_hidden'])
emb_update = str2bool(model_flags['emb_update'])
training_lang = model_flags['training_lang']
learning_rate = model_flags['learning_rate']
nepochs = model_flags['num_epochs']

test_files = FLAGS.test_set.split(',')

# Data Preparation
# ==================================================

if not FLAGS.pre_training:
    assert test_lang == tr_lang
    _, _, voc, dic_inv = unpickle_data(folder)
    test_lex, _, _, test_cue, _, test_y = int_processor.load_test(test_files, voc, scope_dect, event_dect, test_lang)
else:
    test_set, dic_inv, pre_emb_w, _ = ext_processor.load_test(test_files, scope_dect, event_dect, test_lang, emb_size, POS_emb)
    test_lex, _, _, test_cue, _, test_y = test_set

if FLAGS.pre_training: vocsize = pre_emb_w.shape[0]
else: vocsize = len(voc['w2idxs'])

# Evaluation
# ==================================================

def feeder(_bilstm, lex, cue, tags, _y, train = True):
    X = padding(lex, FLAGS.max_sent_length, vocsize - 1)
    C = padding(cue, FLAGS.max_sent_length, 2)
    if tags: T = padding(tags, FLAGS.max_sent_length, tag_voc_size - 1)
    Y = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],_y)).astype('int32'),FLAGS.max_sent_length,0,False)
    _mask = [1 if t!=vocsize - 1 else 0 for t in X]
    feed_dict={
        _bilstm.x: X,
        _bilstm.c: C,
        _bilstm.y: Y,
        _bilstm.istate_fw: numpy.zeros((1, 2*FLAGS.num_hidden)),
        _bilstm.istate_bw: numpy.zeros((1, 2*FLAGS.num_hidden)),
        _bilstm.seq_len: numpy.asarray([len(lex)]),
        _bilstm.mask: _mask}
    if tags: feed_dict.update({_bilstm.t:T})
    if train:
        feed_dict.update({_bilstm.lr:clr})
        _, acc_train = sess.run([optimizer, bi_lstm.accuracy], feed_dict = feed_dict)
        return acc_train
    else:
        acc_test, pred = sess.run([bi_lstm.accuracy,bi_lstm.label_out], feed_dict = feed_dict)
        return acc_test, pred , Y


graph = tf.Graph()
with graph.as_default():
    # session_conf = tf.ConfigProto(
    #   allow_soft_placement=allow_soft_placement,
    #   log_device_placement=log_device_placement)
    sess = tf.Session()
    with sess.as_default():
        bi_lstm = BiLSTM(
                num_hidden=num_hidden,
                num_classes=num_classes,
                voc_dim=vocsize,
                emb_dim=embedding_dim,
                sent_max_len = max_sent_length,
                tag_voc_dim = tag_voc_size,
                tags = True if POS_emb in [1,2] else False,
                external = pre_training,
                update = emb_update)
        saver = tf.train.Saver(tf.all_variables())
        # saver.restore(sess, checkpoint_file)
        # print "Model restored!"

        # load model from last checkpoint
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess,checkpoint_file)
        print "Model restored!"
        # Collect the predictions here
        test_tot_acc = []
        preds_test, gold_test = [],[]
        for i in xrange(len(test_lex)):
            if POS_emb in [1,2]:
                acc_test, pred_test, Y_test = feeder(bi_lstm, test_lex[i], test_cue[i], test_tags[i] if POS_emb == 1 else test_tags_uni[i],test_y[i], train = False)
            else:
                acc_test, pred_test, Y_test = feeder(bi_lstm, test_lex[i], test_cue[i],test_y[i], train = False)
            test_tot_acc.append(acc_test)
            # get prediction softmax
            preds_test.append(pred_test[:len(test_lex[i])])
            gold_test.append(Y_test[:len(test_lex[i])])
        print 'Mean test accuracy: ', sum(test_tot_acc)/len(test_lex)
        _,report_tst,best_test = get_eval(preds_test,gold_test)

        write_report(folder,report_tst,best_test,FLAGS.test_name)
        store_prediction(folder, test_lex, dic_inv, preds_test, gold_test,FLAGS.test_name)

        # # Collect the predictions here
        # test_tot_acc = []
        # pred_test, gold_test = [],[]

        # for i in xrange(len(test_lex)):
        #     X = padding(test_lex[i],max_sent_length,voc_size - 1)
        #     C = padding(test_cue[i],max_sent_length,2)
        #     if POS_emb == 1:
        #         T = padding(test_tags[i],max_sent_length,tag_voc_size - 1)
        #     if POS_emb == 2:
        #         T = padding(test_tags_uni[i],max_sent_length,tag_voc_size - 1)
        #     Y = padding(np.asarray(map(lambda x: [1,0] if x == 0 else [0,1],test_y[i])).astype('int32'),max_sent_length,0,False)
        #     _mask = [1 if _t!=voc_size else 0 for _t in X]
        #     feed_dict = {
        #         bi_lstm.x: X,
        #         bi_lstm.c: C,
        #         bi_lstm.y: Y,
        #         bi_lstm.istate_fw: np.zeros((1, 2*num_hidden)),
        #         bi_lstm.istate_bw: np.zeros((1, 2*num_hidden)),
        #         bi_lstm.seq_len: np.asarray([len(test_lex[i])]),
        #         bi_lstm.mask: _mask
        #         }
        #     if POS_emb in [1,2]:
        #         feed_dict.update({bi_lstm.t: T})
        #     pred, acc = sess.run([bi_lstm.label_out,bi_lstm.accuracy], feed_dict = feed_dict)

        #     test_tot_acc.append(get_accuracy(acc,len(test_lex[i])))

        #     # get prediction softmax
        #     pred_test.append(pred[:len(test_lex[i])])
        #     gold_test.append(Y[:len(test_lex[i])])
        # print 'Mean test accuracy: ', sum(test_tot_acc)/len(test_lex)
        # _,report_tst,best_test = get_eval(pred_test,gold_test)


        # print "Storing reports..."
        # with codecs.open(os.path.join(FLAGS.checkpoint_dir,'%s_report.txt' % FLAGS.test_name),'wb','utf8') as store_rep_test:
        #     store_rep_test.write(report_tst)
        #     store_rep_test.write(str(best_test)+"\n")
        # print "Reports stored..."

        # print "Storing labelling results for dev set..."
        # with codecs.open(os.path.join(FLAGS.checkpoint_dir,'best_%s.txt' % FLAGS.test_name),'wb','utf8') as store_pred:
        #     for s, y_sys, y_hat in zip(test_lex,pred_test,gold_test):
        #         s = [voc_inv['idxs2w'][w] if w in voc_inv['idxs2w'] else '<UNK>' for w in s]
        #         assert len(s)==len(y_sys)==len(y_hat)
        #         for _word,_sys,gold in zip(s,y_sys,y_hat):
        #             _p = list(_sys).index(_sys.max())
        #             _g = 0 if list(gold)==[1,0] else 1
        #             store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
        #         store_pred.write("\n")