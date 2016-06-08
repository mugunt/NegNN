# -*-coding:utf-8-*-

def create_omission(lex,cues,pos,gold):
	assert len(lex) == len(cues) == len(pos) == len(gold)

	om_lex = [[lex[j] for j in xrange(len(lex)) if j!=i] for i in xrange(lex) if cues[i]!=1]
	om_cues = [[cues[j] for j in xrange(len(cues)) if j!=i]for i in xrange(cues) if cues[i]!=1]
	om_pos = [[pos[j] for j in xrange(len(pos)) if j!=i] for i in xrange(pos) if cues[i]!=1]
	om_gold = [[gold[j] for j in xrange(len(gold)) if j!=i] for i in xrange(gold) if cues[i]!=1] 

	# : return list of lists
	return om_lex, om_cues, om_pos, om_gold



