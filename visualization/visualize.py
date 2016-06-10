# -*-coding:utf-8-*-

class Sentence(list):
    def __init__(self,cue):
        super(Sentence,self).__init__()
        # position of the cue in the Sentence
        self.cue = cue

    def append_omission(self,sent):
        self.append(sent)

    def calculate_pos2cue(self):
        for o in self:
            o.set_pos2cue(min([abs(o.index - c) for c in self.cue]))

class Omission(object):
    def __init__(self,
        cosf,
        cosb,
        tag,
        word,
        index):

        self.cosf = cosf
        self.cosb = cosb
        self.tag = tag
        self.word = word
        self.index = index
        self.pos2cue = -1

    def set_pos2cue(self,value):
        self.pos2cue = value


def create_omission(lex,cues,pos,gold):
    assert len(lex) == len(cues) == len(pos) == len(gold)

    om_lex = [[lex[j] for j in xrange(len(lex)) if j!=i] for i in xrange(len(lex)) if cues[i]!=1]
    om_cues = [[cues[j] for j in xrange(len(cues)) if j!=i]for i in xrange(len(cues)) if cues[i]!=1]
    om_pos = [[pos[j] for j in xrange(len(pos)) if j!=i] for i in xrange(len(pos)) if cues[i]!=1]
    om_gold = [[gold[j] for j in xrange(len(gold)) if j!=i] for i in xrange(len(gold)) if cues[i]!=1] 

    # : return list of lists
    return om_lex, om_cues, om_pos, om_gold

