#!/usr/bin/env python3
"""gets cumulative blue score from scratch"""
import numpy as np
from collections import Counter
import math


#
# def n_gram_generator(sentence, n=2, n_gram=False):
#     # print(sentence)
#     # count sublists to check for 'reference
#     if sum(1 for item in sentence if (isinstance(item, list))):
#             sentence = [item for sublist in sentence for item in sublist]
#     # lowercase_sentences = [item.lower() for item in sentence]
#     sent_arr = np.array(sentence)
#     length = len(sent_arr)
#
#     word_list = []
#     for i in range(length+1):
#         if i < n:
#             continue
#         word_range = list(range(i-n, i))
#         s_list = sent_arr[word_range]
#         string = ''.join(s_list)
#         word_list.append(string)
#         if n_gram:
#             word_list = list(set(word_list))
#     return word_list
#
# def cumulative_bleu(references, sentence, n):
#     """gets cumulative blue"""
#     mt_length = len(sentence)
#     o_length = len(references)
#
#     # Brevity Penality
#     if mt_length > o_length:
#         BP = 1
#     else:
#         penality = 1-(mt_length/o_length)
#         BP = np.exp(penality)
#
#     clipped_precision_score = []
#     for i in range(1, n):
#         original_n_gram = Counter(n_gram_generator(references, i))
#         machine_n_gram = Counter(n_gram_generator(sentence, i))
#
#         c = sum(machine_n_gram.values())
#         for j in machine_n_gram:
#             if j in original_n_gram:
#                 if machine_n_gram[j] > original_n_gram[j]:
#                     machine_n_gram[j] = original_n_gram[j]
#                 else:
#                     machine_n_gram[j] = 0
#
#         clipped_precision_score.append(sum(machine_n_gram.values())/ c)
#
#     weights = [0.25] * 4
#     s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
#     s = BP * math.exp(math.fsum(s))
#     return s




def ngrams(input_list, n):
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

        if abs(ref_length - len(sentence_ngrams)) < abs(best_reference_length - len(sentence_ngrams)):
            best_reference_length = ref_length
            best_overlap = overlap

    precision = best_overlap / len(sentence_ngrams) if len(sentence_ngrams) > 0 else 0
    brevity_penalty = math.exp(1 - best_reference_length / len(sentence_ngrams)) if len(
        sentence_ngrams) < best_reference_length else 1
    return brevity_penalty * precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references: List of reference translations (list of lists of words).
        sentence: Candidate sentence (list of words).
        n: Maximum n-gram size to use for evaluation.

    Returns:
        The cumulative BLEU score.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")

    # Compute BLEU score for each n-gram level from 1 to n
    bleu_scores = []
    for i in range(1, n + 1):
        score = ngram_bleu(references, sentence, i)
        bleu_scores.append(score)

    # Calculate geometric mean of the scores
    geometric_mean = math.exp(sum(math.log(score) for score in bleu_scores if score > 0) / n) if all(
        score > 0 for score in bleu_scores) else 0

    # Brevity penalty for cumulative score
    reference_lengths = [len(ref) for ref in references]
    best_reference_length = min(reference_lengths, key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len))
    brevity_penalty = math.exp(1 - best_reference_length / len(sentence)) if len(
        sentence) < best_reference_length else 1

    # Final cumulative BLEU score
    return brevity_penalty * geometric_mean
