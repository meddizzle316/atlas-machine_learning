#!/usr/bin/env python3
"""calculates sensitivity of matrix"""
import numpy as np


def sensitivity(confusion):
    """given confusion, calculates sensitivity"""
    true_positive = []
    false_negative = []
    sensitivity = []

    columns = confusion.shape[1]
    rows = confusion.shape[0]
    for i in range(rows):
        true_positive.append(confusion[i, i])
        row_fp = 0
        for x in range(columns):
            if i != x:
                row_fp += confusion[i, x]
        false_negative.append(row_fp)

    for true_positive, false_negative in zip(true_positive, false_negative):
        class_sensitivity = true_positive / (true_positive + false_negative)
        sensitivity.append(float(f"{class_sensitivity:.8}"))
    return np.array(sensitivity)
