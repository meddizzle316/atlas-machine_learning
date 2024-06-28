#!/usr/bin/env python3
"""cnn forward pass without tensorflow"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """cnn forward pass without tf"""
    
    # by activation function, does that mean we have to provide the function? 
    # or will that be provided for us
    
    m = A_prev.shape[0] # 50,000
    h_prev = A_prev.shape[1] # 28
    w_prev = A_prev.shape[2] # 28
    c_prev = A_prev.shape[3] # 1

    # print("A check for m", m)
    # print("h_prev", h_prev)
    # print("w_prev", w_prev)
    # print("c_prev", c_prev)

    kh = W.shape[0] # output 3
    kw = W.shape[1] # output 3
    kc_prev = W.shape[2] # output 1, is c_prev? 

    # kh is a kernel, at least the printed output is a 3d matrix
    # kw is a kernel, 3d matrix
    # c_new is a kernel I think, 3d matrix
    # print("c_prev", c_prev) # documentation is wrong, only 3 in W
    # I'm assuming that it's the c_prev?

    # print("This is b,", b) 
    # output [[[[ 0.3130677  -0.85409574]]]]
    # does this mean that c_new is 2 ? 
    c_new = b.shape[3]
    # print("this is c_new", c_new)

    stride_h, stride_w = stride
    
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    if padding == 'same':
        pad_h =  kh // 2
        pad_w = kw // 2

    output_height = int(((h_prev + (2 * pad_h) - kh) / stride_h) + 1)
    output_width = int(((w_prev + (2 * pad_w) - kw) / stride_w) + 1)
    
    padded_prev_A = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0) ))
    
    # print("this is the output_height", output_height) 26 if valid, 28 if same
    # print("this is the output width", output_width) 26 if valid, 28 if same

    output = np.zeros((m, output_height, output_width, c_new)) # feel like this 4th dimension is gonna get me

    new_b = np.squeeze(b)
    # print("This is the shape of new_b", new_b.shape) (2, )
    # print("this is new_b", new_b) [ 0.3130677  -0.85409574]
    # print(b[...,0])
    # print(b[...,1])

    for ch in range(c_new):
        for i in range(output_height):
            for j in range(output_width):
                output[:, i, j, ch] += activation(np.sum(((padded_prev_A[:, i*stride_h: i*stride_h + kh, j*stride_w:j*stride_w + kw, :] * W[..., ch])), axis=(1, 2, 3)) ) + new_b[ch]

    return output
