#!/usr/bin/env python3
"""computes simple policy"""
import numpy as np


def softmax(x):
    """performs softmax function as numpy doesn't have a built in one"""
    logits = np.exp(x - np.max(x))
    return logits / np.sum(logits, axis=1, keepdims=True)


def policy(matrix, weight):
    """computes policy with a weight of a matrix. You just matmul and then
    apply softmax"""
    base = np.matmul(matrix, weight)

    policy_softmax = softmax(base)

    return policy_softmax
