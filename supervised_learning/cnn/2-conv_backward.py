#!/usr/bin/env python3
"""convolutional network backward without tf"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """conv backward without tensorflow"""

    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape

    kh, kw, _, _ = W.shape

    # W is (3, 3, 1, 2)
    # print("this is h_new", h_new)
    # print("this is h_prev", h_prev)
    # print("this is c_prev", c_prev)

    sh, s_w = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        # ph =  round(((sh - 1) * h_prev - sh + kh) / 2)
        # pw = round(((s_w - 1) * w_prev - s_w + kw) / 2)
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2) + 1
        pw = ((((w_prev - 1) * s_w) + kw - w_prev) // 2) + 1

    # print("this is ph", ph)
    # print("this is pw", pw)
    # dA_prev = np.matmul(W[:, :,...])
    da = np.zeros(A_prev.shape)
    # da = np.zeros_like(W)
    dW = np.zeros(W.shape)

    # where is the learning rate? Oh we're just getting the derivatives
    # not updating the weights
    # print("this is the shape of W", W.shape) # Output  (3, 3, 1, 2)
    # print("this is the shape of dz", dZ.shape) # Output (10, 26, 26, 2)
    # print('this is the shape of da', da.shape) # Output (10, 28, 28, 1)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    # why don't you divide by the number of samples (10)

    # print(dZ[0, 0, 0, 0])

    # W_expand = np.expand_dims(W, axis=0)
    # W_expand = np.repeat(W_expand, 28, axis=0)
    # W_expand = np.expand_dims(W_expand, axis=0)
    # W_expand = np.repeat(W_expand, 28, axis=0) size (28, 28, 3, 3, 1, 2)
    # print("This is the shape of W_expand", W_expand.shape)
    # (28, 28, 3, 3, 1, 2)
    # # print("this is W_expand", W_expand)

    # print(h_new / sh)

    # print("this is the shape of A_prev", A_prev.shape) # Output (10, 28, 28, 1)

    W_expand = np.expand_dims(W, axis=0)
    W_expand = np.repeat(W_expand, 28, axis=0)
    W_expand = np.expand_dims(W_expand, axis=0)
    W_expand = np.repeat(W_expand, 28, axis=0) #size (28, 28, 3, 3, 1, 2)
    # print("This is the shape of W_expand", W_expand.shape)
    # print("this is W_expand", W_expand)

    # print("this is A_prev", A_prev)

    A_prev_expand = np.expand_dims(A_prev, axis=-1)
    A_prev_expand = np.repeat(A_prev_expand, 2, axis=-1)
    A_prev_expand = np.expand_dims(A_prev_expand, axis=-3)
    A_prev_expand = np.repeat(A_prev_expand, 3, axis=-3)
    A_prev_expand = np.expand_dims(A_prev_expand, axis=-4)
    A_prev_expand = np.repeat(A_prev_expand, 3, axis=-4)

    # print("this is the shape of A_prev_expand", A_prev_expand.shape)
    # print("this is A_prev_expand", A_prev_expand)

    # result = A_prev_expand[1, :, :, :, :, :, :] * W_expand

    # print("this is the shape of dz", dZ.shape) # Output (10, 26, 26, 2)


    p_da = np.pad(da, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    # print("this is the shape of da after padding", da.shape)
    # print("this is the shape of A_prev after padding", A_prev.shape)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):

                    p_da[n, i * sh:i * sh + kh, j * s_w:j * s_w + kw, :] += (
                        W[:, :, :, c] * dZ[n, i, j, c])

                    dW[:, :, :, c] += (
                        A_prev[n, i * sh:i * sh + kh, j * s_w:j * s_w + kw, :]
                        * dZ[n, i, j, c])

    if padding == 'same':
        da = p_da[:, ph:-ph, pw:-pw, :]
    else:
        da = p_da
    return da, dW, db

