#!/usr/bin/env python3
"""does unigram BLeu score"""
from collections import Counter
import math


def uni_bleu(references, sentence):
    """calculates unigram BLEU score"""
    best_overlap = 0
    best_reference_length = float('inf')

    for reference in references:
        ref_count = Counter(reference)
        sentence_count = Counter(sentence)

        overlap = sum((ref_count & sentence_count).values())
        ref_length = len(reference)

        if abs(ref_length - len(sentence)) < abs(best_reference_length - len(sentence)):
            best_reference_length = ref_length
            best_overlap = overlap

    precision = best_overlap / len(sentence) if len(sentence) > 0 else 0

    brevity_penalty = math.exp(1 - best_reference_length / len(sentence)) if len(sentence) < best_reference_length else 1

    return (brevity_penalty * precision)
