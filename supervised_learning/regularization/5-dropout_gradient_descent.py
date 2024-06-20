#!/usr/bin/env python3
"""dropout gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """dropout gradient descent"""

    # getting total number of 'examples'
    m = Y.shape[1]

    dz = cache[f"A{L}"] - Y
    for layer in range(L, 0, -1):
        # going backwards through layers
        # a = cache[f"A{layer - 1}"]
        dw = np.matmul(dz, cache[f"A{layer - 1}"].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if layer > 1:
            # dz is this every iteration besides first
            # getting da of previuos layer
            activation_prime = (1 - np.square(cache[f"A{layer - 1}"]))
            da = np.matmul(weights[f"W{layer}"].T, dz)

            # getting mask of previous layer
            d = cache[f"D{layer-1}"]

            # getting dz with main difference being the mask applied at here
            dz = da * d * activation_prime
            # I think the main difference for dropout is we just multiply
            # 'mask'
            # of the previous layer is applied to dz, effectively setting
            # gradients
            # of those to 0
            dz /= keep_prob
            # correcting the value to get expected value despite changing
            # the gradient

        # dynamically calculates derivatives in each layer

        # update weights dictionary (weights and bias)
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        weights[f'W{layer}'] = W - (alpha * dw)
        weights[f'b{layer}'] = b - (alpha * db)
