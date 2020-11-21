# Evaluates the PCFG
from nltk.parse.viterbi import ViterbiParser
from nltk.corpus import treebank
import pickle
import sys

subset_begin = int(sys.argv[1])
subset_end = int(sys.argv[2])
assert subset_end >= subset_begin

# load unk grammar
pcfg_unk = pickle.load(open("grammar_unk.pcfg", 'rb'))

# use unk grammar on test sentences
parser = ViterbiParser(pcfg_unk)

# test sentences
test = treebank.fileids()[190:]
test_sentence = []
test_pos = []
for sentence in treebank.parsed_sents(test):
    test_sentence.append(sentence.leaves())
    test_pos.append(sentence.pos())

def parse(parser: ViterbiParser, sentence):
    for tree in parser.parse(sentence):
        yield tree        

# predict trees
def get_predictions():
    for sentence in test_sentence[subset_begin:subset_end]:
        for tree in parse(parser, sentence):
            yield tree.pos()

# evaluate
correct = 0
total = 0
for predicted, truth in zip(get_predictions(), test_pos[subset_begin: subset_end]):
    this_sentence_correct = 0
    this_sentence_total = 0
    
    for (word_predicted, pos_predicted), (word, pos) in zip(predicted, truth):
        if word_predicted != word:
            raise ValueError(f"Leaves of the trees are different, prediction word: {word_predicted} vs real word: {word}")
        
        if pos_predicted == pos:
            this_sentence_correct += 1
        this_sentence_total += 1

    # sentence stats
    print("Sentence length: {}".format(len(predicted)))
    print("Unknown count: {}".format(len([_ for _, pos in predicted if pos == 'UNK'])))
    print("--Sentence accuracy: {}".format(this_sentence_correct/this_sentence_total if this_sentence_total != 0 else 0))

    correct += this_sentence_correct
    total += this_sentence_total

print("Total accuracy {}".format(correct/total))