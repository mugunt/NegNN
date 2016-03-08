'''
A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
# import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
from is13.utils.tools_tf import shuffle, minibatch_same_size,padding
from sklearn import metrics
import numpy
import cPickle
import random
import codecs
import os,sys
import time
import subprocess

'''
To classify images using a bidirectional reccurent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
'''

def load(fname):
    with open(fname,'rb') as f:
        train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def random_uniform(shape,name,low=-1.0,high=1.0):
    return  tf.Variable(0.2 * tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32),name=name)

def get_accuracy(p,len_sent):
    return float(len([a for a in p[:len_sent] if a]))/float(len_sent)

def get_eval(predictions,gs):
    y,y_ = [],[]
    for p in predictions: y.extend(map(lambda x: list(x).index(x.max()),p))
    for g in gs: y_.extend(map(lambda x: 0 if list(x)==[1,0] else 1,g))

    print metrics.classification_report(y_,y)
    cm = metrics.confusion_matrix(y_,y)
    print cm

    p,r,f1,s =  metrics.precision_recall_fscore_support(y_,y)
    report = "%s\n%s\n%s\n%s\n\n" % (str(p),str(r),str(f1),str(s)) 

    return numpy.average(f1,weights=s),report,cm

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('-p',help="Pickled fname containing training,test and dev data")
    parser.add_argument('-f',help="Folder to store the log and best system")
    parser.add_argument('-t',action="store_true", help="Use normal POS tags")
    parser.add_argument('-u',action="store_true", help="Use Universal POS tags")
    args = parser.parse_args()

    params = {'clr':1e-4,
        'nhidden':200, # number of hidden units
        'seed':345,
        'es':50, # dimension of word embedding
        'nepochs':100,
        'logf':args.f}

    # store results for epoch
    folder = os.path.join("/Users/ffancellu/git/is13/log/birnn",params['logf'])
    if not os.path.exists(folder): os.mkdir(folder)

    train_set, valid_set, test_set, dic = load(args.p)
    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    if args.t:
        idx2tag = dict((k,v) for v,k in dic['tags2idx'].iteritems())
    if args.u:
        idx2tag = dict((k,v) for v,k in dic['tags_uni2idxs'].iteritems())

    train_lex, train_tags, train_tags_uni,train_y, train_cue, train_scope = train_set
    valid_lex, valid_tags, valid_tags_uni, valid_y, valid_cue, valid_scope = valid_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)
    if args.t:
        ntags = len(dic['tags2idx']) if args.t else 0
    if args.u:
        ntags = len(dic['tags_uni2idxs']) if args.u else 0
    display_step = 100

    # instanciate the model
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])

    nh = params['nhidden']
    nc = nclasses

    # Parameters
    training_iters = 50
    # we fix the maximum sent length to an enormous value
    MAX_SENT_LEN = 100
    emb_size = params['es']
    # Network Parameters
    n_hidden = 200 # hidden layer num of features
    n_classes = 2 # MNIST total classes (0-9 digits)

    # tf Graph
    seq_len = tf.placeholder(tf.int64)
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32)
    c = tf.placeholder(tf.int32)
    t = tf.placeholder(tf.int32)
    mask = tf.placeholder("float")
    # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
    istate_fw = tf.placeholder("float", [None, 2*n_hidden])
    istate_bw = tf.placeholder("float", [None, 2*n_hidden])
    y = tf.placeholder("float", [None, n_classes])


    # Define weights
    _weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        'w_emb' : random_uniform([vocsize+1,emb_size],'w_emb'),
        'c_emb' : random_uniform([3,emb_size],'c_emb'),
        't_emb' : random_uniform([ntags+1,emb_size],'t_emb'),
        'hidden_w': tf.Variable(tf.random_normal([emb_size, 2*n_hidden])),
        'hidden_c': tf.Variable(tf.random_normal([emb_size, 2*n_hidden])),
        'hidden_t': tf.Variable(tf.random_normal([emb_size, 2*n_hidden])),
        'out_w': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    _biases = {
        'hidden_b': tf.Variable(tf.random_normal([2*n_hidden])),
        'out_b': tf.Variable(tf.random_normal([n_classes]))
    }

    def BiRNN(_X, _C, _T, _istate_fw, _istate_bw, _weights, _biases):
        # input: a [len_sent,len_seq] (e.g. 7x5)
        # transform into embeddings
        emb_x = tf.nn.embedding_lookup(_weights['w_emb'],_X)
        emb_c = tf.nn.embedding_lookup(_weights['c_emb'],_C)
        emb_t = tf.nn.embedding_lookup(_weights['t_emb'],_T)

        # Linear activation
        _X = tf.matmul(emb_x, _weights['hidden_w']) + tf.matmul(emb_c,_weights['hidden_c']) + tf.matmul(emb_t,_weights['hidden_t']) + _biases['hidden_b']

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0,MAX_SENT_LEN,_X)

        # Get lstm cell output
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,initial_state_fw = _istate_fw, initial_state_bw=_istate_bw,sequence_length = seq_len)

        return outputs
        # Linear activation
        # Get inner loop last output

    # pred = BiRNN(x, c, istate_fw, istate_bw, _weights, _biases)
    pred = BiRNN(x, c, t, istate_fw, istate_bw, _weights, _biases)

    last_y = [tf.matmul(item, _weights['out_w']) + _biases['out_b'] for item in pred]
    final_outputs = tf.squeeze(tf.pack(last_y))

    # Define loss and optimizer
    cost = tf.reduce_sum(tf.mul(tf.nn.softmax_cross_entropy_with_logits(final_outputs, y),mask))/tf.reduce_sum(mask) # softmax
    # # loss
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost) # Adam Optimizer

    normalize_w_emb = tf.nn.l2_normalize(_weights['w_emb'],1)
    normalize_c_emb = tf.nn.l2_normalize(_weights['c_emb'],1)
    normalize_t_emb = tf.nn.l2_normalize(_weights['t_emb'],1)

    ax = tf.nn.softmax(final_outputs)

    correct_pred = tf.equal(tf.argmax(ax,1), tf.argmax(y,1))

    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Launch the graph
    all_train_acc = []
    all_dev_acc = []
    best_f1 = 0.0
    params['dry'] = 0
    with tf.Session() as sess:
        sess.run(init)
        try:
            step = 1
            for e in xrange(params['nepochs']):
                # shuffle
                shuffle([train_lex,train_y,train_cue,train_tags, train_tags_uni], params['seed'])
                print '[learning] epoch %d' % e
                mean_acc = 0.0
                params['ce'] = e
                tic = time.time()
                for i in xrange(nsentences):
                    X = minibatch_same_size(train_lex[i],MAX_SENT_LEN,vocsize)[-1]
                    C = minibatch_same_size(train_cue[i],MAX_SENT_LEN,2)[-1]
                    if args.t:
                        T = minibatch_same_size(train_tags[i],MAX_SENT_LEN,ntags)[-1]
                    if args.u:
                        T = minibatch_same_size(train_tags_uni[i],MAX_SENT_LEN,ntags)[-1]
                    Y = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],train_y[i])).astype('int32'),MAX_SENT_LEN,0,False)
                    _mask = [1 if _t!=vocsize else 0 for _t in X]
                    sess.run(optimizer, feed_dict={
                        x: X,
                        c: C,
                        y: Y,
                        t: T,
                        istate_fw: numpy.zeros((1, 2*n_hidden)),
                        istate_bw: numpy.zeros((1, 2*n_hidden)),
                        seq_len: numpy.asarray([len(train_lex[i])]),
                        mask: _mask,
                        lr: params['clr']
                        })
                    sess.run(normalize_w_emb)
                    sess.run(normalize_c_emb)
                    sess.run(normalize_t_emb)
                    # Calculating batch accuracy
                    p = sess.run(correct_pred, feed_dict={
                        x: X,
                        c: C,
                        y: Y,
                        t: T,
                        istate_fw: numpy.zeros((1, 2*n_hidden)),
                        istate_bw: numpy.zeros((1, 2*n_hidden)),
                        seq_len: numpy.asarray([len(train_lex[i])]),
                        mask: _mask
                        })
                    mean_acc += get_accuracy(p,len(train_lex[i]))
                    
                    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()
                    
                print "TRAINING MEAN ACCURACY: ", mean_acc/nsentences
                all_train_acc.append(mean_acc/nsentences)

                mean_dev_acc = 0.0
                predictions_dev = []
                gold_ys_dev = []

                all_sents_acc_DEV = []
                for i in xrange(len(valid_lex)):
                    # pad the word and cue vector of the dev set with random shit
                    X_dev = minibatch_same_size(valid_lex[i],MAX_SENT_LEN,vocsize)[-1]
                    C_dev = minibatch_same_size(valid_cue[i],MAX_SENT_LEN,2)[-1]
                    if args.t:
                        T_dev = minibatch_same_size(valid_tags[i],MAX_SENT_LEN,ntags)[-1]
                    if args.u:
                        T_dev = minibatch_same_size(valid_tags_uni[i],MAX_SENT_LEN,ntags)[-1]
                    Y_dev = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],valid_y[i])).astype('int32'),MAX_SENT_LEN,0,False)
                    _mask_dev = [1 if t_d!=vocsize else 0 for t_d in X_dev]

                    # get dev set accuracy
                    dev_p = sess.run(correct_pred, feed_dict={
                        x: X_dev,
                        c: C_dev,
                        y: Y_dev,
                        t: T_dev,
                        istate_fw: numpy.zeros((1, 2*n_hidden)),
                        istate_bw: numpy.zeros((1, 2*n_hidden)),
                        seq_len: numpy.asarray([len(valid_lex[i])]),
                        mask: _mask_dev
                        })
                    mean_dev_acc += get_accuracy(dev_p,len(valid_lex[i]))
                    all_sents_acc_DEV.append(get_accuracy(dev_p,len(valid_lex[i])))
                    # get prediction softmax
                    pred_softmax = sess.run(ax,feed_dict={
                        x: X_dev,
                        c: C_dev,
                        y: Y_dev,
                        t: T_dev,
                        istate_fw: numpy.zeros((1, 2*n_hidden)),
                        istate_bw: numpy.zeros((1, 2*n_hidden)),
                        seq_len: numpy.asarray([len(valid_lex[i])]),
                        mask: _mask_dev
                        })
                    predictions_dev.append(pred_softmax[:len(valid_lex[i])])
                    gold_ys_dev.append(Y_dev[:len(valid_lex[i])])
                print 'DEV MEAN ACCURACY: ',mean_dev_acc/len(valid_lex)
                all_dev_acc.append(mean_dev_acc/len(valid_lex))
                f1,rep_dev,cm_dev = get_eval(predictions_dev,gold_ys_dev)

                if f1 > best_f1:

                    best_f1 = f1
                    print "Best f1 is: ",best_f1
                    params['be'] = params['ce']

                    print "Removing content of folder..."
                    for _f in os.listdir(folder):
                        subprocess.call(['rm', os.path.join(folder,_f)])

                    # store the weights
                    saver.save(sess, os.path.join(folder,"model.ckpt"))
                    print "Model saved."

                    print "Storing accuracy for each sentence..."
                    numpy.save(os.path.join(folder,'all_sents_acc_dev_acc'),numpy.asarray(all_sents_acc_DEV))
                    print "Accuracy for each sentence stored."

                    print "Storing reports..."
                    with codecs.open(os.path.join(folder,'valid_report.txt'),'wb','utf8') as store_rep_dev:
                        store_rep_dev.write(rep_dev)
                        store_rep_dev.write(str(cm_dev)+"\n")
                    print "Reports stored..."

                    print "Storing predictions..."
                    with codecs.open(os.path.join(folder,'best_valid.txt'),'wb','utf8') as store_pred:
                        for s, y_sys, y_hat in zip(valid_lex,predictions_dev,gold_ys_dev):
                            s = [idx2word[w] for w in s]
                            assert len(s)==len(y_sys)==len(y_hat)
                            for _word,_sys,gold in zip(s,y_sys,y_hat):
                                _p = list(_sys).index(_sys.max())
                                _g = 0 if list(gold)==[1,0] else 1
                                store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
                            store_pred.write("\n")
                    params['dry'] = 0
                else:
                    params['dry'] += 1

                if abs(params['be']-params['ce']) >= 10 and params['dry']>=5:
                    print "Halving the lr..."
                    params['clr'] *= 0.5
                    params['dry'] = 0
        except(KeyboardInterrupt):
            pass
        print "Storing all epochs accuracies..."
        numpy.save(os.path.join(folder,'train_acc'),numpy.asarray(all_train_acc))
        numpy.save(os.path.join(folder,'dev_acc'),numpy.asarray(all_dev_acc))