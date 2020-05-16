###################################################
# Image Captioning with Deep Reinforcement Learning
# SJSU CMPE-297-03 | Spring 2020
#
#
# Team:
# Pratikkumar Prajapati
# Aashay Mokadam
# Karthik Munipalle
###################################################

"""
Computes the BLEU, ROUGE, METEOR, and CIDER
using the COCO metrics scripts
"""
import os
import sys

# this requires the coco-caption package, https://github.com/tylin/coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def load_text_data(filename):
    """
    ## Code taken from https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py and made further changes
    """
    contents_file = open(filename, "r")
    contents = []
    for x in contents_file:
        d = " ".join([w for w in x.split(' ') if
                      ('<END>' not in w and '<START>' not in w and '<UNK>' not in w and '\n' not in w)])
        contents.append(d)
    return contents


def load_textfiles(reference_file, hypothesis_file):
    """
    ## Code taken from https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py and made further changes
    """
    references = load_text_data(reference_file)
    hypothesis = load_text_data(hypothesis_file)
    # print("The number of references is {}".format(len(references)))
    refs = {idx: [lines.strip()] for (idx, lines) in enumerate(references)}
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    #     raw_refs = [map(str.strip, r) for r in zip(*references)]
    #     refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs", len(hypo), len(refs))
    return refs, hypo


def score(ref, hypo):
    """
    ## Code taken from https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py and made further changes
    """

    # block prints
    sys.stdout = open(os.devnull, 'w')
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    # enable print
    sys.stdout = sys.__stdout__
    return final_scores


def get_singleton_score(reference, hypothesis):
    refs = {0: [reference.strip()]}
    hypo = {0: [hypothesis.strip()]}
    return score(refs, hypo)
