#!/usr/bin/env python3
"""gradient descent with l2 reg cost"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """gradient descent with l2 reg cost"""
    
    m = Y.shape[1]
    # m = 50,000

    for i in range(L, 0, -1):


        # getting derivatives for that layer

        if i == L:
            dz = cache[f"A{L}"] - Y
        else:
            dz = da * (1 - (cache[f"A{i}"]) ** 2)

        da = np.matmul(weights[f"W{i}"].T, dz) 
        dw = np.matmul(dz, cache[f"A{i - 1}"].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        W = weights[f"W{i}"]
        b =  weights[f"b{i}"]
        # adding L2 reg

        l2_dw = dw + ((lambtha / m) * W) 
        # l2_db = db + ((lambtha / m) * W)
        # updating the weights (do this all at one time)
        weights[f"W{i}"] = W - (alpha * l2_dw)
        weights[f"b{i}"] = b - (alpha * db) 

