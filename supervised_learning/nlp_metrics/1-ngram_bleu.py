#!/usr/bin/env python3
"""does unigram BLeu score"""
from collections import Counter
import math


def ngrams(sequence, n):
    """Generate all n-grams from the given sequence of tokens."""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates the BLEU score for a specific n-gram level."""
    if not references or not sentence:
        return 0.0

    sentence_ngrams = list(ngrams(sentence, n))
    if not sentence_ngrams:
        return 0.0

    sentence_count = Counter(sentence_ngrams)

    best_overlap = 0
    best_reference_length = float('inf')

    for reference in references:
        if not reference:
            continue

        reference_ngrams = list(ngrams(reference, n))
        if not reference_ngrams:
            continue

        ref_count = Counter(reference_ngrams)

        overlap = sum((ref_count & sentence_count).values())
        ref_length = len(reference_ngrams)

        if abs(
                ref_length -
                len(sentence_ngrams)) < abs(
                best_reference_length -
                len(sentence_ngrams)):
            best_reference_length = ref_length
            best_overlap = overlap

    precision = best_overlap / \
        len(sentence_ngrams) if len(sentence_ngrams) > 0 else 0.0
    brevity_penalty = (math.exp(
        1 - best_reference_length /
        len(sentence_ngrams))
        if len(sentence_ngrams) < best_reference_length else 1.0)
    return brevity_penalty * precision
