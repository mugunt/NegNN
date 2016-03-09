from itertools import islice

import random
import numpy as np

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win, flag, idx=None):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    if flag == "cue":
        lpadded = win/2 * [0] + l + win/2 * [0]
    else:
        lpadded = win/2 * [idx] + l + win/2 * [idx]
    out = [lpadded[i:i+win] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def contextwin_lr(l, win_left, win_right, flag, idx=None):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''

    l = list(l)

    if flag == "cue":
        lpadded = win_left * [0] + l + win_right * [0]
    else:
        lpadded = win_left * [idx] + l + win_right * [idx]
    # print lpadded
    out = [lpadded[i:i+win_right+win_left+1] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def minibatch_same_size(l, max_len, pad_idx):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    return np.asarray([np.concatenate((l[:i],[pad_idx]*(max_len-i))) for i in range(1,len(l)+1)])

def padding(l,max_len,pad_idx,x=True):
    if x: pad = [pad_idx]*(max_len-len(l))
    else: pad = [[0,1]]*(max_len-len(l))
    return np.concatenate((l,pad),axis=0)

if __name__=="__main__":
    for l in contextwin_lr([23,43,5,6,890,71],23,31,'word',0):
        print l,l[23]