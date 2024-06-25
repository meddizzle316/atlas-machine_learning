#!/usr/bin/env python3
"""convolve gray scale"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """convolution with numpy"""
    m = images.shape[0]
    # cross correlation
    # kernel = np.flipud(np.fliplr(kernel))
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    out_height = int((images.shape[1] - kh) + 1 / 1)
    out_width = int((images.shape[2] - kw) + 1 / 1)

    output = np.zeros((m, out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            output[:, i, j] += np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
            # calculates the sum over the 2nd and third dimensions, ignoring the first

    return np.array(output)
