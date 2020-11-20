from typing import List, Set
from nltk.corpus import treebank
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction, induce_pcfg
from nltk.parse.viterbi import ViterbiParser
import pickle

from nltk.tree import Tree

def print_tree(set_of_sentences):
    for tree in treebank.parsed_sents(set_of_sentences):
        print(tree)

def induce_grammar(train):
    """Induces a PCFG from the given set of Penn Treebank sentences

    Args:
        train (Any): Set of Penn Treebank sentences

    Returns:
        PCFG: A PCFG grammar instance
    """
    productions = []
    for item in train:
      for tree in treebank.parsed_sents(item):
        # perform optional tree transformations, e.g.:
        tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()

    S = Nonterminal('S')
    return induce_pcfg(S, productions)

def get_missing_words(grammar: PCFG, sentences: List[Tree]):
    """Get words that are not in the grammar but in the given sentences

    Args:
        grammar (PCFG): Induced grammar on the train set
        sentences (List[Tree]): List of Penn Treebank sentences in Tree form

    Returns:
        set[str] : List of unique missing words
    """

    missing_words = []
    for tree in treebank.parsed_sents(sentences):
        tokens = tree.leaves()
        missing_words += [tok for tok in tokens if not grammar._lexical_index.get(tok)]

    return set(missing_words)

def fill_missing_words(grammar : PCFG, missing_words: Set[str]):
    unknown = Nonterminal('UNK')
    rule = ProbabilisticProduction(unknown, list(missing_words), prob=1)
    adjusted_productions = grammar.productions()
    adjusted_productions.append(rule)
    return PCFG(Nonterminal('S'), adjusted_productions)

def main():
    # train = treebank.fileids()[:190]
    test = treebank.fileids()[190:] # 10 last sentences

    # # original grammar
    # pcfg = induce_grammar(train)
    # pickle.dump(pcfg, open("grammar.pcfg", 'wb'))

    # load grammar
    pcfg : PCFG = pickle.load(open("grammar.pcfg", 'rb'))

    # fill in missing words
    missing_words = get_missing_words(pcfg, test)
    pcfg_unk = fill_missing_words(pcfg, missing_words)

    pickle.dump(pcfg_unk, open("grammar_unk.pcfg", 'wb'))

if __name__ == "__main__":
    main()