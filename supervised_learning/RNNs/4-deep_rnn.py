#!/usr/bin/env python3
"""forward pass on deep rnn in numpy"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep rnn"""
    layers = len(rnn_cells)
    T, m, i = X.shape  # 6, 8, 10
    h = h_0.shape[2]
    H = np.empty((T + 1, layers, m, h))
    H[0] = h_0
    Y = np.empty((T, m, 5))
    for t in range(T):
        x_t = X[t]
        for layer in range(layers):
            if t == 0:
                h_1 = h_0[0]
                h_2 = h_0[1]
                h_3 = h_0[2]

                if layer == 0:
                    h_prev_0, y1 = rnn_cells[layer].forward(h_1, x_t)
                    H[t + 1, layer] = h_prev_0
                if layer == 1:
                    h_prev_1, y2 = rnn_cells[layer].forward(h_2, h_prev_0)
                    H[t + 1, layer] = h_prev_0
                if layer == 2:
                    h_prev_2, y3 = rnn_cells[layer].forward(h_3, h_prev_1)
                    H[t + 1, layer] = h_prev_0
                    Y[0] = y3
            else:
                if layer == 0:
                    h_prev_0, y1 = rnn_cells[layer].forward(h_prev_0, x_t)
                    H[t + 1, layer] = h_prev_0
                if layer == 1:
                    h_prev_1, y2 = rnn_cells[layer].forward(
                        h_prev_1, h_prev_0)
                    H[t + 1, layer] = h_prev_1
                if layer == 2:
                    h_prev_2, y3 = rnn_cells[layer].forward(
                        h_prev_2, h_prev_1)
                    H[t + 1, layer] = h_prev_2
                    Y[t] = y3
    return H, Y
