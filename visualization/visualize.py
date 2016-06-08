# -*-coding:utf-8-*-

class Sentence(list):
    def __init__(self,cue):
        super(Sentences,self).__init__()
        # position of the cue in the Sentence
        self.cue = cue

    def append_omission(self,sent):
        self.append(sent)

    def calculate_pos2cue(self):
        for o in self:
            o.set_pos2cue(min([abs(o.pos - c) for c in self.cue]))

class Omission(object):
    def __init__(self,
        cosf,
        cosb,
        tag,
        word,
        pos):

        self.cosf = cosf
        self.cosb = cosb
        self.tag = tag
        self.word = word
        self.pos = pos
        self.pos2cue = -1

    def set_pos2cue(self,value):
        self.pos2cue = value


def create_omission(lex,cues,pos,gold):
    assert len(lex) == len(cues) == len(pos) == len(gold)

    om_lex = [[lex[j] for j in xrange(len(lex)) if j!=i] for i in xrange(lex) if cues[i]!=1]
    om_cues = [[cues[j] for j in xrange(len(cues)) if j!=i]for i in xrange(cues) if cues[i]!=1]
    om_pos = [[pos[j] for j in xrange(len(pos)) if j!=i] for i in xrange(pos) if cues[i]!=1]
    om_gold = [[gold[j] for j in xrange(len(gold)) if j!=i] for i in xrange(gold) if cues[i]!=1] 

    # : return list of lists
    return om_lex, om_cues, om_pos, om_gold

