#!/usr/bin/env python3
"""convolve gray scale"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """convolution with numpy"""
    m = images.shape[0]
    # cross correlation
    # kernel = np.flipud(np.fliplr(kernel))
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    s = 1
    p = round((s -1 ) *images.shape[1] + kh / 2)
    # print(f"this is the padding {p}")


    out_height = int((images.shape[1] + (p)- kh) + 1)
    out_width = int((images.shape[2] + (p)- kw) + 1)

    # print("this is out height", out_height)
    # print("this is out width", out_width)
    output = np.zeros((m, out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            padded_input = np.pad(images, ((0,0), (kernel.shape[0] // 2, kernel.shape[0] // 2),
                                           (kernel.shape[1]//2, kernel.shape[1] // 2)), mode='constant')
            
            output[:, i, j] += np.sum(padded_input[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
            # calculates the sum over the 2nd and third dimensions, ignoring the first

    return np.array(output)
