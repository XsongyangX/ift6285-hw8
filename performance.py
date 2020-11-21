# Evaluates the PCFG
from nltk.parse.viterbi import ViterbiParser
from nltk.corpus import treebank
import pickle

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
subset = 2
predictions = [tree.pos() for tree in parse(parser, test_sentence[:subset])]

# evaluate
correct = 0
total = 0
for predicted, truth in zip(predictions, test_pos):

    for (word_predicted, pos_predicted), (word, pos) in zip(predicted, truth):
        if word_predicted != word:
            raise ValueError(f"Leaves of the trees are different, prediction word: {word_predicted} vs real word: {word}")
        
        if pos_predicted == pos:
            correct += 1
        total += 1

    # sentence stats
    print("Sentence length: {}".format(len(predicted)))
    print("Unknown count: {}".format(len([_ for _, pos in predicted if pos == 'UNK'])))
    print("Sentence accuracy: {}".format(correct/total if total != 0 else 0))

print("Total accuracy {}".format(correct/total))