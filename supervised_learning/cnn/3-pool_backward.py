#!/usr/bin/env python3
"""backprop in pooling from scratch"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """backprop in pooling layer"""

    # print(dA.shape) # output (10, 9, 9, 2)
    # print(A_prev.shape) # (10, 28, 28, 2)
    # print(kernel_shape) # (3, 3)
    # print(stride) # (3, 3)
    # print(mode)

    da_prev = np.zeros(dA.shape)

    kh, kw = kernel_shape
    m, h_new, w_new, c_new = dA.shape
    m, h_x, w_x, c_prev = A_prev.shape
    sh, sw = stride

    da_prev = np.zeros_like(A_prev)  

    for n in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    if mode == 'max':
                        temp = A_prev[n, h*sh:h*sh+kh, w*sw:w*sw+kw, ch]
                        mask = (temp == np.max(temp))
                        da_prev[n, h*sh:h*sh+kh, w*sw:w*sw+kw, ch] += dA[n, h, w, ch] * mask
                    if mode == 'avg':
                        da_prev[n, h*sh:h*sh+kh, w*sw:w*sw+kw, ch] += (dA[m, h, w, ch]) /kh/kw
    return da_prev
