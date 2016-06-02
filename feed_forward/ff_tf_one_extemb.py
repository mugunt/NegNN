from argparse import ArgumentParser
from is13.utils.tools_tf import shuffle, minibatch, contextwin_lr
from sklearn import metrics
from gensim.models import Word2Vec

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

    args = parser.parse_args()

    params = {'clr':1e-4,
        'nhidden':200, # number of hidden units
        'seed':345,
        'es':50, # dimension of word embedding
        'nepochs':50,
        'win_left':9, # number of words in the context window to the left
        'win_right':16, # number of words in the context window to the right
        'w_syn0':'/Users/ffancellu/git/is13/w2v_models/we_lm_sh_data_new_50/VectorModel-we_lm_sh_data_50.data.syn0.npy',
        'ext_i2w':'/Users/ffancellu/git/is13/w2v_models/we_lm_sh_data_new_50/index2word.npy',
        'logf':args.f}

    folder = os.path.join("/Users/ffancellu/git/is13/log/feedforw",params['logf'])
    if not os.path.exists(folder): os.mkdir(folder)

    train_set, valid_set, test_set, dic = load(args.p)

    # load word embedding ext model
    we_model = numpy.load(params['w_syn0'])
    w_syn0 = pad_embeddings(we_model,params['es'])
    ext_i2w = numpy.load(params['ext_i2w'])

    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    
    # create a mapping for the word indices between internal and external dicts
    word2idx_w2v = dict([(w,i) for i,w in enumerate(ext_i2w)])
    idx2idx_w = get_idx_mapping_w(idx2word,word2idx_w2v,w_syn0.shape[0])

    train_lex, _, _, train_y, train_cue, train_scope = train_set
    valid_lex, _, _, valid_y, valid_cue, valid_scope = valid_set

    train_lex = transform(train_lex,idx2idx_w)
    valid_lex = transform(valid_lex,idx2idx_w)

    vocsize = len(word2idx_w2v)-1
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    word2idx_w2v = dict([(v,k) for k,v in word2idx_w2v.iteritems()])
    word2idx_w2v.update({"<UNK>":w_syn0.shape[0]-2})

    # instanciate the model
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])

    nh = params['nhidden']
    nc = nclasses

    with tf.device('/cpu:0'):
        word_emb = tf.Variable(0.2 * tf.random_uniform([w_syn0.shape[0],params['es']], minval=-1.0, maxval=1.0, dtype=tf.float32),name='w_emb')
        cue_emb = random_uniform([2,params['es']],'c_emb')
    Wx = random_uniform([params['es']*(params['win_left']+params['win_right']+1),nh],'Wx')
    Wc = random_uniform([params['es']*(params['win_left']+params['win_right']+1),nh],"Wc")
    W = random_uniform([nh,nc],"W")
    b = tf.Variable(tf.zeros([nh]),'b')
    bo = tf.Variable(tf.zeros([nc]),'bo')

    # setting the variables
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, params['win_left']+params['win_right']+1], name='input_x')
    c = tf.placeholder(tf.int32, shape=[None, params['win_left']+params['win_right']+1], name='input_c')
    y_ = tf.placeholder(tf.float32, shape=[None, nc], name='input_y')

    rsh_x = tf.reshape(x,[-1])
    rsh_c = tf.reshape(c,[-1])
    emb_x = tf.nn.embedding_lookup(word_emb,rsh_x)
    emb_c = tf.nn.embedding_lookup(word_emb,rsh_c)
    emb_x_rsh = tf.reshape(emb_x,[-1,params['es']*(params['win_left']+params['win_right']+1)])
    emb_c_rsh = tf.reshape(emb_c,[-1,params['es']*(params['win_left']+params['win_right']+1)])

    y = tf.nn.softmax(tf.matmul(tf.sigmoid(tf.matmul(emb_x_rsh,Wx) + tf.matmul(emb_c_rsh,Wc) + b), W) + bo)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    normalize_w_emb = tf.nn.l2_normalize(word_emb,1)
    normalize_c_emb = tf.nn.l2_normalize(cue_emb,1)

    # for testing purposes
    prediction = tf.argmax(y,1)
    g = y_
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())
    sess.run(word_emb.assign(w_syn0))

    # train with early stopping on validation set
    all_train_acc = []
    all_dev_acc = []
    all_test_acc = []

    best_f1 = 0.0
    params['dry'] = 0
    try:
        for e in xrange(params['nepochs']):
            shuffle([train_lex,train_y,train_cue], params['seed'])
            print '[learning] epoch %d' % e
            params['ce'] = e
            tic = time.time()
            train_acc = 0.0
            for i in xrange(nsentences):
                cwords = contextwin_lr(train_lex[i], params['win_left'], params['win_right'],'word',vocsize)
                ccues = contextwin_lr(train_cue[i],params['win_left'], params['win_right'],'cue')
                labels = map(lambda x: numpy.asarray([1,0]).astype('int32') if x ==0 else numpy.asarray([0,1]).astype('int32'),train_y[i])
                sess.run(train_step, feed_dict={x: cwords, c: ccues, y_: labels,lr: params['clr']})    
                train_acc += sess.run(accuracy, feed_dict={x: cwords, c: ccues, y_: labels})
                sess.run(normalize_w_emb)
                sess.run(normalize_c_emb)
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            print 'DEV mean accuracy: ',train_acc/len(train_lex)
            all_train_acc.append(train_acc/nsentences)

            # DEV SET
            dev_acc = 0.0
            predictions_dev = []
            gold_ys_dev = []
            all_sents_acc_DEV = []
            for i in xrange(len(valid_lex)):
                cwords_dev = contextwin_lr(valid_lex[i], params['win_left'], params['win_right'],'word',vocsize)
                ccues_dev = contextwin_lr(valid_cue[i],params['win_left'], params['win_right'],'cue')
                labels_dev = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),valid_y[i])
                acc = sess.run(accuracy, feed_dict={x: cwords_dev, c: ccues_dev, y_: labels_dev})
                dev_acc += acc
                all_sents_acc_DEV.append(acc)
                prediction = y.eval(feed_dict={x: cwords_dev, c: ccues_dev, y_: labels_dev},session=sess)
                predictions_dev.append(prediction)
                gold_ys_dev.append(labels_dev)
            current_ma = dev_acc/len(valid_lex)
            print 'DEV mean accuracy: ',dev_acc/len(valid_lex)
            dev_f1,rep_dev,cm_dev = get_eval(predictions_dev,gold_ys_dev)
            all_dev_acc.append(dev_acc/len(valid_lex))

            if dev_f1 > best_f1:
                best_f1 = dev_f1
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

                print "Storing labelling results for dev set..."
                with codecs.open(os.path.join(folder,'best_valid.txt'),'wb','utf8') as store_pred:
                    for s, y_sys, y_hat in zip(valid_lex,predictions_dev,gold_ys_dev):
                        s = [word2idx_w2v[w] for w in s]
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
