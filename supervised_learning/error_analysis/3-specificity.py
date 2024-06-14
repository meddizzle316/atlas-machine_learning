#!/usr/bin/env python3
"""module for specificity"""
import numpy as np


def specificity(confusion):
    """gets specificity of confusion"""
    sum = np.sum(confusion, axis=0)
    true_positive = np.diag(confusion)
    false_positve = sum - true_positive
    true_negative = []
    sum_rows = np.sum(confusion, axis=1)
    for i in range(len(confusion)):
        sum_class_row = np.sum(confusion[i])
        sum_cl_column = sum[i] - true_positive[i]
        true_negative.append(np.sum(confusion) - sum_class_row - sum_cl_column)
    return true_negative / (true_negative + false_positve)
