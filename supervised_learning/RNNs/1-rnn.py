#!/usr/bin/env python3
"""makes simple rnn unit"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """does forward propagation for simple rnn unit"""

    t, m, i = X.shape
    h_prev = h_0.copy()

    for i in range(t):

        h_prev, y = rnn_cell.forward(h_prev, X[i])
        if i == 0:
            h_all = np.concatenate((h_0, h_prev), axis=0)
            y_all = y
        else:
            h_all = np.concatenate((h_all, h_prev), axis=0)
            y_all = np.concatenate((y_all, y), axis=0)
    h_all = h_all.reshape(-1, m, 15)
    y_all = y_all.reshape(-1, m, 5)
    return h_all, y_all
