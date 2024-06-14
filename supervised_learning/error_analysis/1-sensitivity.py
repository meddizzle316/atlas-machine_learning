#!/usr/bin/env python3
"""calculates sensitivity of matrix"""


def sensitivity(confusion):
    """given confusion, calculates sensitivity"""
    true_positive = []
    false_positive = []
    sensitivity = []

    # getting true and false positives
    columns = confusion.shape[1]
    rows = confusion.shape[0]
    for i in range(rows):
        true_positive.append(confusion[i, i])
        row_fp = 0
        for x in range(columns):
            if i != x:
                row_fp += confusion[i, x]
        false_positive.append(row_fp)

    # getting sensitivity
    # I should probably have something like assert len(true_positive)
    # == len(false_positive) but whatever
    for true_positive, false_positive in zip(true_positive, false_positive):
        class_sensitivity = true_positive / (true_positive + false_positive)
        sensitivity.append(class_sensitivity)
    return sensitivity
