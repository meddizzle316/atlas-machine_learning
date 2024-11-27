#!/usr/bin/env python3
"""gets cumulative blue score from scratch"""
import numpy as np
from collections import Counter
import math


def ngrams(input_list, n):
    """function to get n-gram"""
    return zip(*[input_list[i:] for i in range(n)])


def ngram_bleu(references, sentence, n):
    """Calculates the BLEU score for a specific n-gram level."""
    sentence_ngrams = list(ngrams(sentence, n))
    sentence_count = Counter(sentence_ngrams)

    best_overlap = 0
    best_reference_length = float('inf')

    for reference in references:
        reference_ngrams = list(ngrams(reference, n))
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
        len(sentence_ngrams) if len(sentence_ngrams) > 0 else 0
    brevity_penalty = (math.exp(
        1 - best_reference_length /
        len(sentence_ngrams))
        if len(sentence_ngrams) < best_reference_length else 1)
    return brevity_penalty * precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")

    bleu_scores = []
    for i in range(1, n + 1):
        score = ngram_bleu(references, sentence, i)
        bleu_scores.append(score)

    geometric_mean = math.exp(
        sum(
            math.log(score) for score in bleu_scores if score > 0) /
        n) if all(
        score > 0 for score in bleu_scores) else 0

    reference_lengths = [len(ref) for ref in references]
    best_reference_length = min(reference_lengths, key=lambda ref_len: (
        abs(ref_len - len(sentence)), ref_len))
    brevity_penalty = math.exp(
        1 -
        best_reference_length /
        len(sentence)) if len(sentence) < best_reference_length else 1

    return brevity_penalty * geometric_mean
