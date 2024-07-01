#!/usr/bin/env python3
"""pooling forward without tf"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """pooling forward without tf"""
    sh, sw = stride
    kh, kw = kernel_shape
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # print("this is m", m)
    # print("this is h_prev", h_prev)
    # print("this is w_prev", w_prev)
    # print("this is c_prev", c_prev)

    out_height = ((h_prev - kh) // sh + 1)
    out_width = ((w_prev - kw) // sw + 1)

    output = np.zeros((m, out_height, out_width, c_prev))

    for i in range(out_height):
        for j in range(out_width):
            if mode == 'max':
                output[:, i, j, :] += np.max(
                    A_prev[:, i*sh: i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] += np.mean(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
    return output
