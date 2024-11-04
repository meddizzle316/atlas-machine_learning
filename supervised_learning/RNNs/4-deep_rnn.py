#!/usr/bin/env python3
"""forward pass on deep rnn in numpy"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep rnn"""
    l = len(rnn_cells)
    T, m, i = X.shape # 6, 8, 10
    #h_0 3, 8, 15
    # what if the 3 is for the first timestep?

    for t in range(T):
        x_t = X[t]
        for layer in range(l):
            if t == 0:
                h_1 = h_0[0]
                h_2 = h_0[1]
                h_3 = h_0[2]

                match layer:
                    case 0:
                        h_prev_0, y1 = rnn_cells[layer].forward(h_1, x_t)
                    case 1:
                        h_prev_1, y2 = rnn_cells[layer].forward(h_2, h_prev_0)
                    case 2:
                        h_prev_2, y3 = rnn_cells[layer].forward(h_3, h_prev_1)
                        print("shape h_0", h_0.shape)
                        H = np.concatenate((h_prev_0, h_prev_1, h_prev_2), axis=-1)
                        # H = H.reshape((3, 8, 15))

                        print("shape H", H.shape)
                        H_t = np.concatenate((h_1, h_2, h_3), axis=-1)
                        H = np.concatenate((H_t, H), axis=-1)
                        Y = y3
            else:
                match layer:
                    case 0:
                        h_prev_0, y1 = rnn_cells[layer].forward(h_prev_0, x_t)
                    case 1:
                        h_prev_1, y2 = rnn_cells[layer].forward(h_prev_1, h_prev_0)
                    case 2:
                        h_prev_2, y3 = rnn_cells[layer].forward(h_prev_2, h_prev_1)
                        H_t = np.concatenate((h_prev_0, h_prev_1, h_prev_2), axis=-1)

                        H = np.concatenate((H, H_t), axis=-1)
                        Y = np.concatenate((Y, y3), axis=-1)
    H = H.reshape((-1, l, m, 15))
    Y = Y.reshape((t + 1, m, -1))
    return H, Y


# def deep_rnn(rnn_cells, X, h_0):
#     """Performs forward propagation for a deep RNN."""
#     l = len(rnn_cells)  # number of layers
#     t, m, i = X.shape  # time steps, batch size, input size
#     _, _, h = h_0.shape  # hidden state size
#
#     # Initialize output arrays
#     H = np.zeros((t + 1, l, m, h))
#     Y = []
#     H[0] = h_0  # set initial hidden state
#
#     # Loop over time steps
#     for step in range(t):
#         x_t = X[step]
#         for layer in range(l):
#             h_prev = H[step, layer]
#             if layer == 0:
#                 # For the first layer, input is x_t
#                 h_next, y = rnn_cells[layer].forward(h_prev, x_t)
#             else:
#                 # For subsequent layers, input is the hidden state from the previous layer
#                 h_next, y = rnn_cells[layer].forward(h_prev,  H[step + 1, layer - 1])
#
#             # Update hidden state
#             H[step + 1, layer] = h_next
#
#         # Collect output from the last layer
#         Y.append(y)
#
#     # Convert Y list to numpy array
#     Y = np.array(Y)
#
#     return H, Y


