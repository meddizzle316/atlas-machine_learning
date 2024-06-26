#!/usr/bin/env python3
"""pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """max and average pooling"""
    sh, sw = stride
    kh, kw = kernel_shape
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    out_height = ((h - kh) // sh + 1)
    out_width = ((w - kw) // sw + 1)
    # print("this is the out height", out_height)
    # print("this is the out width", out_width)

    output = np.zeros((m, out_height, out_width, c))
    
    for j in range(out_height):
        for i in range(out_width):
            if mode == 'max':
                output[:, i, j, :] += np.max(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] += np.average(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))

    return output
