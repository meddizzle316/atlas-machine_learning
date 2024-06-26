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

    out_height = h // kh # should be 16
    out_width = w // kw # should be 16

    output = np.zeros((m, out_height, out_width, c))
    
    for i in range(out_height):
        for j in range(out_width):
            if mode == 'max':
                output[:, i, j, :] += np.max(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] += np.mean(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))

    return output
