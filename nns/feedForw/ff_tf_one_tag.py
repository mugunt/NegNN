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

    s = {'clr':1e-4,
        'win_left':9, # number of words in the context window to the left
        'win_right':16, # number of words in the context window to the right
        'bs':20, # avg length of the sentence
        'nhidden':200, # number of hidden units
        'seed':345,
        'es':50, # dimension of word embedding
        'nepochs':200,
        'logf':args.f}

    folder = os.path.join("/Users/ffancellu/git/is13/log/feedforw",s['logf'])
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
    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    nh = s['nhidden']
    nc = nclasses

    word_emb = random_uniform([vocsize+1,s['es']],'word_emb')
    cue_emb = random_uniform([2,s['es']],'cue_emb')
    tag_emb = random_uniform([ntags+1,s['es']],'tag_emb')
    Wx = random_uniform([s['es']*(s['win_left']+s['win_right']+1),nh],'Wx')
    Wc = random_uniform([s['es']*(s['win_left']+s['win_right']+1),nh],"Wc")
    Wt = random_uniform([s['es']*(s['win_left']+s['win_right']+1),nh],"Wt")
    W = random_uniform([nh,nc],"W")
    b = tf.Variable(tf.zeros([nh]),'b')
    bo = tf.Variable(tf.zeros([nc]),'bo')

    # setting the variables
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, s['win_left']+s['win_right']+1], name='input_x')
    c = tf.placeholder(tf.int32, shape=[None, s['win_left']+s['win_right']+1], name='input_c')
    t = tf.placeholder(tf.int32, shape=[None, s['win_left']+s['win_right']+1], name='input_t')
    y_ = tf.placeholder(tf.float32, shape=[None, nc], name='input_y')

    rsh_x = tf.reshape(x,[-1])
    rsh_c = tf.reshape(c,[-1])
    rsh_t = tf.reshape(t,[-1])
    emb_x = tf.nn.embedding_lookup(word_emb,rsh_x)
    emb_c = tf.nn.embedding_lookup(cue_emb,rsh_c)
    emb_t = tf.nn.embedding_lookup(tag_emb,rsh_t)
    emb_x_rsh = tf.reshape(emb_x,[-1,s['es']*(s['win_left']+s['win_right']+1)])
    emb_c_rsh = tf.reshape(emb_c,[-1,s['es']*(s['win_left']+s['win_right']+1)])
    emb_t_rsh = tf.reshape(emb_t,[-1,s['es']*(s['win_left']+s['win_right']+1)])

    y = tf.nn.softmax(tf.matmul(tf.sigmoid(tf.matmul(emb_x_rsh,Wx) + tf.matmul(emb_c_rsh,Wc) + tf.matmul(emb_t_rsh,Wt) + b), W) + bo)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    normalize_w_emb = tf.nn.l2_normalize(word_emb,1)
    normalize_c_emb = tf.nn.l2_normalize(cue_emb,1)
    normalize_t_emb = tf.nn.l2_normalize(tag_emb,1)

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

    training_times = []

    best_ma = 0.0
    s['dry'] = 0
    try:
        for e in xrange(s['nepochs']):
            # shuffle
            shuffle([train_lex,train_y,train_cue,train_tags,train_tags_uni], s['seed'])
            print '[learning] epoch %d' % e
            s['ce'] = e
            tic = time.time()
            train_acc = 0.0
            for i in xrange(nsentences):
                cwords = contextwin_lr(train_lex[i], s['win_left'], s['win_right'],'word',vocsize)
                ccues = contextwin_lr(train_cue[i],s['win_left'], s['win_right'],'cue')
                if args.t:
                    ctags = contextwin_lr(train_tags[i],s['win_left'], s['win_right'],'word',ntags)
                if args.u:
                    ctags = contextwin_lr(train_tags_uni[i],s['win_left'], s['win_right'],'word',ntags)
                labels = map(lambda x: numpy.asarray([1,0]).astype('int32') if x ==0 else numpy.asarray([0,1]).astype('int32'),train_y[i])
                sess.run(train_step, feed_dict={x: cwords, c: ccues, y_: labels,t: ctags, lr: s['clr']})
                train_acc += sess.run(accuracy, feed_dict={x: cwords, c: ccues,t: ctags, y_: labels})
                sess.run(normalize_w_emb)
                sess.run(normalize_c_emb)
                sess.run(normalize_t_emb)

                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            training_times.append(time.time()-tic)
            print 'DEV mean accuracy: ',train_acc/nsentences
            all_train_acc.append(train_acc/nsentences)

            # DEV SET
            dev_acc = 0.0
            all_sents_acc_DEV = []
            predictions_dev = []
            gold_ys_dev = []
            for i in xrange(len(valid_lex)):
                cwords_dev = contextwin_lr(valid_lex[i], s['win_left'], s['win_right'],'word',vocsize)
                ccues_dev = contextwin_lr(valid_cue[i],s['win_left'], s['win_right'],'cue')
                if args.t:
                    ctags_dev = contextwin_lr(valid_tags[i],s['win_left'], s['win_right'],'word',ntags)
                if args.u:
                    ctags_dev = contextwin_lr(valid_tags_uni[i],s['win_left'], s['win_right'],'word',ntags)
                labels_dev = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),valid_y[i])
                sent_accuracy_dev = sess.run(accuracy, feed_dict={x: cwords_dev, c: ccues_dev, t: ctags_dev, y_: labels_dev})
                dev_acc += sent_accuracy_dev
                all_sents_acc_DEV.append(sent_accuracy_dev)
                prediction = y.eval(feed_dict={x: cwords_dev, c: ccues_dev, t: ctags_dev, y_: labels_dev},session=sess)
                predictions_dev.append(prediction)
                gold_ys_dev.append(labels_dev)
            current_ma = dev_acc/len(valid_lex)
            print 'DEV mean accuracy: ',dev_acc/len(valid_lex)
            dev_f1,rep_dev,cm_dev = get_eval(predictions_dev,gold_ys_dev)
            all_dev_acc.append(dev_acc/len(valid_lex))

            tst_acc = 0.0
            all_sents_acc_TST = []
            predictions_test = []
            gold_ys_test = []
            test_lex, test_tags, test_tags_uni, test_y, test_cue, test_scope = test_set[0]
            for i in xrange(len(test_lex)):
                cwords_test = contextwin_lr(test_lex[i], s['win_left'], s['win_right'],'word',vocsize)
                ccues_test = contextwin_lr(test_cue[i], s['win_left'], s['win_right'],'cue')
                if args.t:
                    ctags_tst = contextwin_lr(test_tags[i],s['win_left'], s['win_right'],'word',ntags)
                if args.u:
                    ctags_tst = contextwin_lr(test_tags_uni[i],s['win_left'], s['win_right'],'word',ntags)
                labels_test = map(lambda x: numpy.asarray([1,0]).astype('int32') if x == 0 else numpy.asarray([0,1]).astype('int32'),test_y[i])
                sent_accuracy_tst = sess.run(accuracy, feed_dict={x: cwords_test, c: ccues_test, t: ctags_tst, y_: labels_test})
                tst_acc += sent_accuracy_tst
                all_sents_acc_TST.append(sent_accuracy_tst)
                prediction = y.eval(feed_dict={x: cwords_test, c: ccues_test, t: ctags_tst, y_: labels_test},session=sess)
                predictions_test.append(prediction)
                gold_ys_test.append(labels_test)
            print 'TEST mean accuracy: ',tst_acc/len(test_lex)
            _,rep_tst,cm_tst = get_eval(predictions_test,gold_ys_test)
            all_test_acc.append(tst_acc/len(test_lex))

            if current_ma > best_ma:
                # store the best epoch
                best_ma = current_ma
                s['be'] = s['ce']
                print "BEST MEAN ACCURACY OF %f AT EPOCH %d" % (best_ma,s['be'])

                print "Removing content of folder..."
                for _f in os.listdir(folder):
                    subprocess.call(['rm', os.path.join(folder,_f)])

                print "Saving model..."
                saver.save(sess, os.path.join(folder,"model.ckpt"))
                print "Model saved."

                print "Storing accuracy for each sentence..."
                numpy.save(os.path.join(folder,'all_sents_acc_dev_acc'),numpy.asarray(all_sents_acc_DEV))
                numpy.save(os.path.join(folder,'all_sents_acc_dev_tst'),numpy.asarray(all_sents_acc_TST))
                print "Accuracy for each sentence stored."

                print "Storing reports..."
                with codecs.open(os.path.join(folder,'valid_report.txt'),'wb','utf8') as store_rep_dev:
                    store_rep_dev.write(rep_dev)
                    store_rep_dev.write(str(cm_dev)+"\n")
                with codecs.open(os.path.join(folder,'tst_report.txt'),'wb','utf8') as store_rep_tst:
                    store_rep_tst.write(rep_tst)
                    store_rep_tst.write(str(cm_tst)+"\n")

                print "Storing labeling results for DEV set..."
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
    numpy.save(os.path.join(folder,'training_times'),numpy.asarray(training_times))





        