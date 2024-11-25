#!/usr/bin/env python3
"""gets cumulative blue score from scratch"""
import numpy as np
from collections import Counter
import math


def n_gram_generator(sentence, n=2, n_gram=False):
    # print(sentence)
    # count sublists to check for 'reference
    if sum(1 for item in sentence if (isinstance(item, list))):
        # print("is reference list")
        sentence = [item for sublist in sentence for item in sublist]
    # lowercase_sentences = [item.lower() for item in sentence]
    sent_arr = np.array(sentence)
    length = len(sent_arr)

    word_list = []
    for i in range(length+1):
        if i < n:
            continue
        word_range = list(range(i-n, i))
        s_list = sent_arr[word_range]
        string = ''.join(s_list)
        word_list.append(string)
        if n_gram:
            word_list = list(set(word_list))
    return word_list

def cumulative_bleu(references, sentence, n):
    """gets cumulative blue"""
    mt_length = len(sentence)
    o_length = len(references)

    # Brevity Penality
    if mt_length > o_length:
        BP = 1
    else:
        penality = 1-(mt_length/o_length)
        BP = np.exp(penality)

    clipped_precision_score = []
    for i in range(1, 5):
        original_n_gram = Counter(n_gram_generator(references, i))
        machine_n_gram = Counter(n_gram_generator(sentence, i))

        c = sum(machine_n_gram.values())
        for j in machine_n_gram:
            if j in original_n_gram:
                if machine_n_gram[j] > original_n_gram[j]:
                    machine_n_gram[j] = original_n_gram[j]
                else:
                    machine_n_gram[j] = 0

        clipped_precision_score.append(sum(machine_n_gram.values())/ c)

    weights = [0.25] * 4
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    s = BP * math.exp(math.fsum(s))
    return s
