from nltk.parse.viterbi import ViterbiParser
from nltk.corpus import treebank
import pickle
import time

# Benchmarks the speed of the Viterbi parser


def parse(parser: ViterbiParser, sentence):
    start_time = time.time()
    parser.trace(trace=1)
    for tree in parser.parse(sentence):
        print(tree)
        print(
            f"Time elapsed for sentence of length {len(sentence)}: {time.time() - start_time}")


# load unk grammar
pcfg_unk = pickle.load(open("grammar_unk.pcfg", 'rb'))

# use unk grammar on test sentences
parser = ViterbiParser(pcfg_unk)

# one sentence
test = treebank.fileids()[190:]
first_sentence = None
for sentence in treebank.parsed_sents(test):
    first_sentence = sentence.leaves()
    break

for i in range(1,7):
    parse(parser, first_sentence[:i])