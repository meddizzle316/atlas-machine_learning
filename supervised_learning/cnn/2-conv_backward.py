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

    # print("this is pad_h", pad_h)
    # print("this is pad_w", pad_w)
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
    # W_expand = np.repeat(W_expand, 28, axis=0) # size (28, 28, 3, 3, 1, 2)
    # print("This is the shape of W_expand", W_expand.shape) # (28, 28, 3, 3, 1, 2)
    # # print("this is W_expand", W_expand)

    # print(h_new / stride_h)

    pad_da = np.pad(da, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0) ))
    pad_A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    # print("this is the shape of da after padding", da.shape)
    # print("this is the shape of A_prev after padding", pad_A_prev.shape)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):

                    pad_da[n, i*stride_h:i *stride_h +kh, j *stride_w :j *stride_w +kw, :] +=   W[:, :, :, c] * dZ[n, i, j, c] 

                    dW[:, :, :, c] += pad_A_prev[n, i*stride_h:i *stride_h +kh, j *stride_w :j *stride_w +kw, :] * dZ[n, i, j, c]
                    # except ValueError:
                    #     print("this is the current shape of da", da[n, i*stride_h:i *stride_h +kh, j *stride_w :j *stride_w +kw, :].shape)
                        

    if padding == 'same':
        da = pad_da[:, pad_h:-pad_h, pad_w:-pad_w, :]
    else:
        da = pad_da
    return da, dW, db
