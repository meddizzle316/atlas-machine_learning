#!/usr/bin/env python3
"""cnn forward pass without tensorflow"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """cnn forward pass without tf"""

    # by activation function, does that mean we have to provide the function?
    # or will that be provided for us

    m, h_prev, w_prev, c_prev = A_prev.shape

    kh, kw, kc_prev, _ = W.shape

    c_new = b.shape[3]

    s_h, s_w = stride

    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    if padding == 'same':
        pad_h = round(((s_h - 1) * h_prev - s_h + kh) / 2)
        pad_w = round(((s_w - 1) * w_prev - s_w + kw) / 2)

    output_height = int(((h_prev + (2 * pad_h) - kh) / s_h) + 1)
    output_width = int(((w_prev + (2 * pad_w) - kw) / s_w) + 1)

    padded_prev_A = np.pad(A_prev, ((0, 0),
                                    (pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    output = np.zeros((m, output_height, output_width, c_new))

    new_b = np.squeeze(b)
    # print("This is the shape of new_b", new_b.shape) (2, )
    # print("this is new_b", new_b) [ 0.3130677  -0.85409574]

    for ch in range(c_new):
        for i in range(output_height):
            for j in range(output_width):
                output[:, i, j, ch] += np.sum((
                    (padded_prev_A[:, i*s_h: i*s_h + kh,
                                   j*s_w:j*s_w + kw, :] * W[:, :, :, ch])),
                                   axis=(1, 2, 3))

    return activation(output + b)
