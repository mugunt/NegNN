from argparse import ArgumentParser
from is13.utils.tools_tf import shuffle, minibatch, contextwin_lr
from sklearn import metrics

import tensorflow as tf
import numpy
import cPickle
import random
import time
import sys,os
import codecs
import subprocess

def pad_embeddings(w2v_we,emb_size):
    # add at <UNK> random vector at -2
    numpy.append(w2v_we,0.2 * numpy.random.uniform(-1.0, 1.0,emb_size))
    # add a padding random vector at -1
    numpy.append(w2v_we,0.2 * numpy.random.uniform(-1.0, 1.0,emb_size))
    return w2v_we

def get_idx_mapping_w(idx2word,word2idx_w2v,dim):
    idx2idx = {}
    for k in idx2word:
        if idx2word[k]=="<UNK>" or idx2word[k].lower() not in word2idx_w2v:
            idx2idx[k] = dim - 2
        else:
            idx2idx[k] = word2idx_w2v[idx2word[k].lower()]
    idx2idx[-1] = dim - 1
    return idx2idx

def get_idx_mapping_t(idx2word,word2idx_w2v,dim):
    idx2idx = {}
    for k in idx2word:
        if idx2word[k]=="<UNK>" or idx2word[k] not in word2idx_w2v:
            idx2idx[k] = dim - 2
        else:
            idx2idx[k] = word2idx_w2v[idx2word[k]]
    idx2idx[-1] = dim - 1
    return idx2idx

def load(fname):
    with open(fname,'rb') as f:
        train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def random_uniform(shape,name,low=-1.0,high=1.0):
    return tf.Variable(0.2 * tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32),name=name)

def transform(m,idx2idx):
    return numpy.asarray([numpy.asarray([idx2idx[i] for i in r]) for r in m])

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
        'win_left':9, # number of words in the context window to the left
        'win_right':16, # number of words in the context window to the right
        'w_syn0':'/Users/ffancellu/git/is13/w2v_models/we_lm_sh_data_new_50/VectorModel-we_lm_sh_data_50.data.syn0.npy',
        'ext_i2w':'/Users/ffancellu/git/is13/w2v_models/we_lm_sh_data_new_50/index2word.npy',
        'logf':args.f}

    if args.t:
        params.update({
            't_syn0':'/Users/ffancellu/git/is13/w2v_models/pos_50_t2v/pos_50_syn0.npy',
            'ext_i2t':'/Users/ffancellu/git/is13/w2v_models/pos_50_t2v/index2word.npy'
            })
    if args.u:
        params.update({
            't_syn0':'/Users/ffancellu/git/is13/w2v_models/uni_50_t2v/uni_50_syn0.npy',
            'ext_i2t':'/Users/ffancellu/git/is13/w2v_models/uni_50_t2v/index2word.npy'
            })

    folder = os.path.join("/Users/ffancellu/git/is13/log/feedforw",params['logf'])
    if not os.path.exists(folder): os.mkdir(folder)

    train_set, valid_set, test_set, dic = load(args.p)

    # load word embedding ext model
    we_model = numpy.load(params['w_syn0'])
    w_syn0 = pad_embeddings(we_model,params['es'])
    ext_i2w = numpy.load(params['ext_i2w'])
    # load tag embedding ext model
    t_model = numpy.load(params['t_syn0'])
    t_syn0 = pad_embeddings(t_model,params['es'])
    ext_i2t = numpy.load(params['ext_i2t'])

    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    if args.t:
        idx2tag = dict((k,v) for v,k in dic['tags2idx'].iteritems())
    if args.u:
        idx2tag = dict((k,v) for v,k in dic['tags_uni2idxs'].iteritems())

    # create a mapping for the word indices between internal and external dicts
    word2idx_w2v = dict([(w,i) for i,w in enumerate(ext_i2w)])
    idx2idx_w = get_idx_mapping_w(idx2word,word2idx_w2v,w_syn0.shape[0])
    
    # create a mapping for the tags indices between internal and external dicts
    tags2idx_w2v = dict([(_t,i) for i,_t in enumerate(ext_i2t)])
    idx2idx_t = get_idx_mapping_t(idx2tag,tags2idx_w2v,t_syn0.shape[0])

    train_lex, train_tags, train_tags_uni, train_y, train_cue, train_scope = train_set
    valid_lex, valid_tags, valid_tags_uni, valid_y, valid_cue, valid_scope = valid_set

    vocsize = len(word2idx_w2v)-1
    ntags = len(tags2idx_w2v)-1
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    train_lex = transform(train_lex,idx2idx_w)
    valid_lex = transform(valid_lex,idx2idx_w)
    if args.t:
        train_tags = transform(train_tags,idx2idx_t)
        valid_tags = transform(valid_tags,idx2idx_t)
    if args.u:
        train_tags_uni = transform(train_tags_uni,idx2idx_t)
        valid_tags_uni = transform(valid_tags_uni,idx2idx_t)

    idx2word_w2v = dict([(v,k) for k,v in word2idx_w2v.iteritems()])
    idx2word_w2v.update({"<UNK>":w_syn0.shape[0]-2})
    idx2tag_w2v = dict([(v,k) for k,v in tags2idx_w2v.iteritems()])

    # instanciate the model
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])

    nh = params['nhidden']
    nc = nclasses

    with tf.device('/cpu:0'):
        word_emb = tf.Variable(0.2 * tf.random_uniform([w_syn0.shape[0],params['es']], minval=-1.0, maxval=1.0, dtype=tf.float32),name='w_emb')
        tag_emb = tf.Variable(0.2 * tf.random_uniform([t_syn0.shape[0],params['es']], minval=-1.0, maxval=1.0, dtype=tf.float32),name='t_emb')
        cue_emb = random_uniform([2,params['es']],'c_emb')
    Wx = random_uniform([params['es']*(params['win_left']+params['win_right']+1),nh],'Wx')
    Wc = random_uniform([params['es']*(params['win_left']+params['win_right']+1),nh],"Wc")
    Wt = random_uniform([params['es']*(params['win_left']+params['win_right']+1),nh],"Wt")
    W = random_uniform([nh,nc],"W")
    b = tf.Variable(tf.zeros([nh]),'b')
    bo = tf.Variable(tf.zeros([nc]),'bo')

    # setting the variables
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, params['win_left']+params['win_right']+1], name='input_x')
    c = tf.placeholder(tf.int32, shape=[None, params['win_left']+params['win_right']+1], name='input_c')
    t = tf.placeholder(tf.int32, shape=[None, params['win_left']+params['win_right']+1], name='input_t')
    y_ = tf.placeholder(tf.float32, shape=[None, nc], name='input_y')

    rsh_x = tf.reshape(x,[-1])
    rsh_c = tf.reshape(c,[-1])
    rsh_t = tf.reshape(t,[-1])
    emb_x = tf.nn.embedding_lookup(word_emb,rsh_x)
    emb_c = tf.nn.embedding_lookup(cue_emb,rsh_c)
    emb_t = tf.nn.embedding_lookup(tag_emb,rsh_t)
    emb_x_rsh = tf.reshape(emb_x,[-1,params['es']*(params['win_left']+params['win_right']+1)])
    emb_c_rsh = tf.reshape(emb_c,[-1,params['es']*(params['win_left']+params['win_right']+1)])
    emb_t_rsh = tf.reshape(emb_t,[-1,params['es']*(params['win_left']+params['win_right']+1)])

    y = tf.nn.softmax(tf.matmul(tf.sigmoid(tf.matmul(emb_x_rsh,Wx) + tf.matmul(emb_c_rsh,Wc) + tf.matmul(emb_t_rsh,Wt) + b), W) + bo)


    # for testing purposes
    prediction = tf.argmax(y,1)
    g = y_
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    saver.restore(sess, os.path.join(folder,"model.ckpt"))
    tst_acc = 0.0
    all_sents_acc_TST = []
    predictions_test = []
    gold_ys_test = []
    test_lex, test_tags, test_tags_uni, test_y, test_cue, test_scope = test_set[0]
    test_lex = transform(test_lex,idx2idx_w)
    if args.t:
        test_tags = transform(test_tags,idx2idx_t)
    if args.u:
        test_tags_uni = transform(test_tags_uni,idx2idx_t)
    for i in xrange(len(test_lex)):
        cwords_test = contextwin_lr(test_lex[i], params['win_left'], params['win_right'],'word',vocsize)    
        ccues_test = contextwin_lr(test_cue[i], params['win_left'], params['win_right'],'cue')
        if args.t:
            ctags_tst = contextwin_lr(test_tags[i],params['win_left'], params['win_right'],'word',ntags)
        if args.u:
            ctags_tst = contextwin_lr(test_tags_uni[i],params['win_left'], params['win_right'],'word',ntags)
        labels_test = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),test_y[i])
        sent_accuracy_tst = sess.run(accuracy, feed_dict={x: cwords_test, c: ccues_test, t: ctags_tst, y_: labels_test})
        tst_acc += sent_accuracy_tst
        all_sents_acc_TST.append(sent_accuracy_tst)
        prediction = y.eval(feed_dict={x: cwords_test, c: ccues_test, t: ctags_tst, y_: labels_test},session=sess)
        predictions_test.append(prediction)
        gold_ys_test.append(labels_test)

    print 'TEST mean accuracy: ',tst_acc/len(test_lex)
    _,rep_tst,cm_tst = get_eval(predictions_test,gold_ys_test)
    # all_test_acc_.append(tst_acc/len(test_lex))

    print "Storing accuracy for each sentence..."
    numpy.save(os.path.join(folder,'all_sents_acc_tst'),numpy.asarray(all_sents_acc_TST))
    print "Accuracy for each sentence stored."

    print "Storing reports..."
    with codecs.open(os.path.join(folder,'test_report.txt'),'wb','utf8') as store_rep_dev:
        store_rep_dev.write(rep_tst)
        store_rep_dev.write(str(cm_tst)+"\n")
    print "Reports stored..."

    print "Storing labelling results for dev set..."
    with codecs.open(os.path.join(folder,'best_test.txt'),'wb','utf8') as store_pred:
        for s, y_sys, y_hat in zip(test_lex,predictions_test,gold_ys_test):
            s = [idx2word_w2v[w] for w in s]
            assert len(s)==len(y_sys)==len(y_hat)
            for _word,_sys,gold in zip(s,y_sys,y_hat):
                _p = list(_sys).index(_sys.max())
                _g = 0 if list(gold)==[1,0] else 1
                store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
            store_pred.write("\n")