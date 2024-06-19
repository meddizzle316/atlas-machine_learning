#!/usr/bin/env python3
"""dropout layer"""
import numpy as np


def dropout_forward_prop(X, in_weights, L, keep_prob):
    """dropout"""
    cache = {}
    cache["A0"] = X # is X numpy?
    for i in range(1, L + 1):
        W = in_weights[f"W{i}"]
        b = in_weights[f"b{i}"]
        if i != L:
            a = np.tanh(np.dot(W, cache[f"A{i - 1}"]) + b)
            d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            a = np.multiply(a, d)
            a /= keep_prob
            cache[f"D{i}"] = d
        else:
            z = np.dot(W, cache[f"A{i - 1}"]) + b
            t = np.exp(z)
            a = t / np.sum(t, axis=0)
        cache[f"A{i}"] = a
    return cache
