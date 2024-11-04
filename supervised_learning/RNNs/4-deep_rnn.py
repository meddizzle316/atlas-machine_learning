#!/usr/bin/env python3
"""forward pass on deep rnn in numpy"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep rnn"""
    layers = len(rnn_cells)
    T, m, i = X.shape  # 6, 8, 10
    # h_0 3, 8, 15
    # what if the 3 is for the first timestep?

    for t in range(T):
        x_t = X[t]
        for layer in range(layers):
            if t == 0:
                h_1 = h_0[0]
                h_2 = h_0[1]
                h_3 = h_0[2]

                if layer == 0:
                    h_prev_0, y1 = rnn_cells[layer].forward(h_1, x_t)
                if layer == 1:
                    h_prev_1, y2 = rnn_cells[layer].forward(h_2, h_prev_0)
                if layer == 2:
                    h_prev_2, y3 = rnn_cells[layer].forward(h_3, h_prev_1)
                    H = np.concatenate(
                        (h_prev_0, h_prev_1, h_prev_2), axis=-1)

                    H_t = np.concatenate((h_1, h_2, h_3), axis=-1)
                    H = np.concatenate((H_t, H), axis=-1)
                    Y = y3
            else:
                if layer == 0:
                    h_prev_0, y1 = rnn_cells[layer].forward(h_prev_0, x_t)
                if layer == 1:
                    h_prev_1, y2 = rnn_cells[layer].forward(
                        h_prev_1, h_prev_0)
                if layer == 2:
                    h_prev_2, y3 = rnn_cells[layer].forward(
                        h_prev_2, h_prev_1)
                    H_t = np.concatenate(
                        (h_prev_0, h_prev_1, h_prev_2), axis=-1)

                    H = np.concatenate((H, H_t), axis=-1)
                    Y = np.concatenate((Y, y3), axis=-1)
    H = H.reshape((-1, layers, m, 15))
    Y = Y.reshape((t + 1, m, -1))
    return H, Y
