#!/usr/bin/env python3
"""convolutional network backward without tf"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """conv backward without tensorflow"""


    m = dZ.shape[0] # OUtput 10
    h_new = dZ.shape[1] # output 26
    w_new = dZ.shape[2] # output 26
    c_new = dZ.shape[3] # output 2


    h_prev = A_prev.shape[0] # output 10, height of previous layer
    w_prev = A_prev.shape[1] # output 28, width of previous layer
    c_prev = A_prev.shape[2] # output 28, number of channels in prev layer??

    # hmm, h_prev and c_prev might be mixed up

    kh = W.shape[0] # 3
    kw = W.shape[1] # 3

    # W is (3, 3, 1, 2)


    new_b = np.squeeze(b)

    stride_h, stride_w = stride
    
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    if padding == 'same':
        pad_h =  round(((stride_h - 1) * h_prev - stride_h + kh) / 2)
        pad_w = round(((stride_w - 1) * w_prev - stride_w + kw) / 2)

    # dA_prev = np.matmul(W[:, :,...])
    da = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    # where is the learning rate? Oh we're just getting the derivatives 
    # not updating the weights
    print("this is the shape of W", W.shape)
    print("this is the shape of dz", dZ.shape)
    print('this is the shape of da', da.shape)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    # why don't you divide by the number of samples (10)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    da[n, i*stride_h:i *stride_h +kh, j *stride_w :j *stride_w +kw, :] +=   W[:, :, :, c] * dZ[n, i, j, c] 


    return da, dW, db
