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

def load(fname):
    with open(fname,'rb') as f:
        train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def random_uniform(shape,name,low=-1.0,high=1.0):
    return  tf.Variable(0.2 * tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32),name=name)

def get_eval(predictions,gs):
    y,y_ = [],[]
    for p in predictions: y.extend(map(lambda x: list(x).index(x.max()),p))
    for g in gs: y_.extend(map(lambda x: 0 if list(x)==[1,0] else 1,g))

    print metrics.classification_report(y_,y)
    print metrics.confusion_matrix(y_,y)

    p,r,f1,s =  metrics.precision_recall_fscore_support(y_,y)

    return numpy.average(f1,weights=s)

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('-p',help="Pickled fname containing training,test and dev data")
    parser.add_argument('-f',help="Folder to store the log and best system")
    parser.add_argument('-t',help="Add POS tag related info",action='store_true')
    args = parser.parse_args()

    s = {'clr':1e-4,
        'win_left':9, # number of words in the context window to the left
        'win_right':16, # number of words in the context window to the right
        'bs':20, # avg length of the sentence
        'nhidden':200, # number of hidden units
        'seed':345,
        'es':50, # dimension of word embedding
        'nepochs':100,
        'logf':args.f}

    folder = os.path.join("/Users/ffancellu/git/is13/log/feedforw",s['logf'])
    if not os.path.exists(folder): os.mkdir(folder)

    train_set, valid_set, test_set, dic = load(args.p)
    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    if args.t:
        idx2tag = dict((k,v) for v,k in dic['tags2idx'].iteritems())

    train_lex, train_tags, train_y, train_cue, train_scope = train_set
    valid_lex, valid_tags, valid_y, valid_cue, valid_scope = valid_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)
    ntags = len(dic['tags2idx']) if args.t else 0

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    nh = s['nhidden']
    nc = nclasses

    word_emb = random_uniform([vocsize+1,s['es']],'word_emb')
    cue_emb = random_uniform([2,s['es']],'cue_emb')
    Wx = random_uniform([s['es']*(s['win_left']+s['win_right']+1),nh],'Wx')
    Wc = random_uniform([s['es']*(s['win_left']+s['win_right']+1),nh],"Wc")
    W = random_uniform([nh,nc],"W")
    b = tf.Variable(tf.zeros([nh]),'b')
    bo = tf.Variable(tf.zeros([nc]),'bo')

    # setting the variables
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, s['win_left']+s['win_right']+1], name='input_x')
    c = tf.placeholder(tf.int32, shape=[None, s['win_left']+s['win_right']+1], name='input_c')
    y_ = tf.placeholder(tf.float32, shape=[None, nc], name='input_y')

    rsh_x = tf.reshape(x,[-1])
    rsh_c = tf.reshape(c,[-1])
    emb_x = tf.nn.embedding_lookup(word_emb,rsh_x)
    emb_c = tf.nn.embedding_lookup(word_emb,rsh_c)
    emb_x_rsh = tf.reshape(emb_x,[-1,s['es']*(s['win_left']+s['win_right']+1)])
    emb_c_rsh = tf.reshape(emb_c,[-1,s['es']*(s['win_left']+s['win_right']+1)])

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

    # train with early stopping on validation set
    all_train_acc = []
    all_dev_acc = []
    all_test_acc = []

    best_ma = 0.0
    s['dry'] = 0
    try:
        for e in xrange(s['nepochs']):
            # shuffle
            if args.t:
                shuffle([train_lex,train_y,train_cue,train_tags], s['seed'])
            else:
                shuffle([train_lex,train_y,train_cue], s['seed'])
            print '[learning] epoch %d' % e
            s['ce'] = e
            tic = time.time()
            train_acc = 0.0
            for i in xrange(nsentences):
                batch_acc = 0.0
                cwords = contextwin_lr(train_lex[i], s['win_left'], s['win_right'],'word',vocsize)
                words = map(lambda x: numpy.asarray(x).astype('int32'),\
                             minibatch(cwords, s['bs']))
                ccues = contextwin_lr(train_cue[i],s['win_left'], s['win_right'],'cue')
                cues_bs = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(ccues, s['bs']))
                labels = map(lambda x: numpy.asarray([1,0]).astype('int32') if x ==0 else numpy.asarray([0,1]).astype('int32'),train_y[i])
                labels = map(lambda x: numpy.asarray(x).astype('int32'),
                             minibatch(labels, s['bs']))
                for word_batch, cues_batch, label in zip(words, cues_bs,labels):
                    sess.run(train_step, feed_dict={x: word_batch, c: cues_batch, y_: label,lr: s['clr']})
                    sess.run(normalize_w_emb)
                    sess.run(normalize_c_emb)
                    batch_acc+=sess.run(accuracy, feed_dict={x: word_batch, c: cues_batch, y_: label,lr: s['clr']})
                train_acc +=batch_acc/len(words)

                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            print 'DEV mean accuracy: ',train_acc/len(train_lex)
            all_train_acc.append(train_acc/nsentences)

            # DEV SET
            dev_acc = 0.0
            predictions_dev = []
            gold_ys_dev = []
            for i in xrange(len(valid_lex)):
                cwords_dev = contextwin_lr(valid_lex[i], s['win_left'], s['win_right'],'word',vocsize)
                ccues_dev = contextwin_lr(valid_cue[i],s['win_left'], s['win_right'],'cue')
                labels_dev = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),valid_y[i])
                dev_acc += sess.run(accuracy, feed_dict={x: cwords_dev, c: ccues_dev, y_: labels_dev})
                prediction = y.eval(feed_dict={x: cwords_dev, c: ccues_dev, y_: labels_dev},session=sess)
                predictions_dev.append(prediction)
                gold_ys_dev.append(labels_dev)
            current_ma = dev_acc/len(valid_lex)
            print 'DEV mean accuracy: ',dev_acc/len(valid_lex)
            dev_f1 = get_eval(predictions_dev,gold_ys_dev)
            all_dev_acc.append(dev_acc/len(valid_lex))

            tst_acc = 0.0
            predictions_test = []
            gold_ys_test = []
            for i,sub_set in enumerate(test_set):
                test_lex, test_tags, test_y, test_cue, test_scope = sub_set
                for j in xrange(len(test_lex)):
                    cwords_test = contextwin_lr(test_lex[j], s['win_left'], s['win_right'],'word',vocsize)
                    ccues_test = contextwin_lr(test_cue[j], s['win_left'], s['win_right'],'cue')
                    labels_test = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),test_y[j])
                    tst_acc += sess.run(accuracy, feed_dict={x: cwords_test, c: ccues_test, y_: labels_test})
                    prediction = y.eval(feed_dict={x: cwords_test, c: ccues_test, y_: labels_test},session=sess)
                    predictions_test.append(prediction)
                    gold_ys_test.append(labels_test)
                print 'TEST mean accuracy: ',tst_acc/len(test_lex)
                _ = get_eval(predictions_test,gold_ys_test)

            if current_ma > best_ma:
                # store the best epoch
                best_ma = current_ma
                s['be'] = s['ce']
                print "BEST MEAN ACCURACY OF %f AT EPOCH %d" % (best_ma,s['be'])
                saver.save(sess, os.path.join(folder,"model.ckpt"))
                print "Model saved."
                print "Storing labelling results for DEV set..."
                with codecs.open(os.path.join(folder,'best_valid.txt'),'wb','utf8') as store_pred:
                    for sent, y_sys, y_hat in zip(valid_lex,predictions_dev,gold_ys_dev):
                        sent = [idx2word[w] for w in sent]
                        assert len(sent)==len(y_sys)==len(y_hat)
                        for _word,_sys,gold in zip(sent,y_sys,y_hat):
                            _p = list(_sys).index(_sys.max())
                            _g = 0 if list(gold)==[1,0] else 1
                            store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
                        store_pred.write("\n")
                print "Storing labelling results for TEST set..."
                with codecs.open(os.path.join(folder,'best_test.txt'),'wb','utf8') as store_pred:
                    for sent, y_sys, y_hat in zip(test_lex,predictions_test,gold_ys_test):
                        sent = [idx2word[w] for w in sent]
                        assert len(sent)==len(y_sys)==len(y_hat)
                        for _word,_sys,gold in zip(sent,y_sys,y_hat):
                            _p = list(_sys).index(_sys.max())
                            _g = 0 if list(gold)==[1,0] else 1
                            store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
                        store_pred.write("\n")
                s['dry'] = 0
            else:
                s['dry'] += 1

            if abs(s['be']-s['ce']) >= 10 and s['dry']>=5:
                print "Halving the lr..."
                s['clr'] *= 0.5
                s['dry'] = 0
    except(KeyboardInterrupt):
        pass
    print "Storing accuracy for training and development..."
    numpy.save(os.path.join(folder,'train_acc'),numpy.asarray(all_train_acc))
    numpy.save(os.path.join(folder,'dev_acc'),numpy.asarray(all_dev_acc))






        