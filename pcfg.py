import time
from typing import List, Set
from nltk.corpus import treebank
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction, Production, induce_pcfg
from nltk.parse.viterbi import ViterbiParser
import pickle

from nltk.tree import Tree


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
            # Remove branches A-B-C into A-B+C
            tree.collapse_unary(collapsePOS=False)
            # Remove A->(B,C,D) into A->B,C+D->D
            tree.chomsky_normal_form(horzMarkov=2)
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
        missing_words += [
            tok for tok in tokens if not grammar._lexical_index.get(tok)]

    return set(missing_words)


def fill_missing_words(grammar: PCFG, missing_words: Set[str]):
    # UNK -> word1 | word2 | ... | wordN
    unknown = Nonterminal('UNK')
    unk_rules = [ProbabilisticProduction(unknown, missing_word, prob=1.0/len(missing_words)) for missing_word in missing_words]
    
    # Add UNK as a possibility to all rules with strings in the right hand side
    corrected_rules : List[Nonterminal] = []
    rule: ProbabilisticProduction
    for rule in grammar.productions():
        
        # right hand side has a string somewhere
        if any(isinstance(element, str) for element in rule.rhs()):
            
            # rule has already been corrected
            if rule.lhs() in corrected_rules:
                continue
            
            rules_to_rebalance = grammar.productions(lhs=rule.lhs())

            unk_rules.append(ProbabilisticProduction(\
                rule.lhs(), [unknown], prob= 1.0 / (len(rules_to_rebalance) + 1)))
            
            # rebalance probabilities
            rule_to_rebalance : ProbabilisticProduction
            for rule_to_rebalance in rules_to_rebalance:
                unk_rules.append(ProbabilisticProduction(\
                    rule_to_rebalance.lhs(),
                    rule_to_rebalance.rhs(),
                    prob=rule_to_rebalance.prob() / (len(rules_to_rebalance) + 1)
                    ))

            corrected_rules.append(rule.lhs())

    return PCFG(Nonterminal('S'), unk_rules)


def parse_treebank(parser: ViterbiParser, sentences):
    start_time = time.time()
    parser.trace(trace=1)
    for sentence in treebank.parsed_sents(sentences):
        tokens = sentence.leaves()
        for tree in parser.parse(tokens):
            print(tree)
            print(
                f"Time elapsed for sentence of length {len(tokens)}: {time.time() - start_time}")


def main():
    # train = treebank.fileids()[:190]
    test = treebank.fileids()[190:]  # 10 last sentences

    # original grammar
    # pcfg = induce_grammar(train)
    # pickle.dump(pcfg, open("grammar.pcfg", 'wb'))

    # load grammar
    # pcfg : PCFG = pickle.load(open("grammar.pcfg", 'rb'))

    # fill in missing words
    # missing_words = get_missing_words(pcfg, test)
    # pcfg_unk = fill_missing_words(pcfg, missing_words)

    # pickle.dump(pcfg_unk, open("grammar_unk.pcfg", 'wb'))

    # load unk grammar
    pcfg_unk: PCFG = pickle.load(open("grammar_unk.pcfg", 'rb'))

    # use unk grammar on test sentences
    parser = ViterbiParser(pcfg_unk)
    parse_treebank(parser, test)


if __name__ == "__main__":
    main()
