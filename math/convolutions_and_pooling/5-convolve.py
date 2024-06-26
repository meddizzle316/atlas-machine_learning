#!/usr/bin/env python3
"""convolve gray scale"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """convolution with numpy"""
    m = images.shape[0]
    # cross correlation
    # kernel = np.flipud(np.fliplr(kernel))
    h = images.shape[1]
    w = images.shape[2]
    ic = images.shape[3] # is '3' for 5-main

    kh = kernels.shape[0]
    kw = kernels.shape[1]
    kc = kernels.shape[2] # 3 for 5-main
    nc = kernels.shape[3] # 3 for 5-main
    
    # print("this is the stride", stride)
    if isinstance(padding, str) and padding == "same":
        pad_h = kernels.shape[0] // 2
        pad_w = kernels.shape[1] // 2
        # print(pad_h)
        # print(pad_w)
        out_height = int((images.shape[1] + (2 * (pad_h))- kh)/stride[0]) + 1
        out_width = int((images.shape[2] + (2 * (pad_w))- kw)/stride[1]) + 1
        padded_input = np.pad(images, ((0,0), (pad_h, pad_h),
                                           (pad_w, pad_w), (0, 0)))
    elif padding == "valid":
        out_height = int((images.shape[1] - kh)/ stride[0]) + 1
        out_width = int((images.shape[2] - kw)/ stride[1]) + 1

        padded_input = images
    elif isinstance(padding, tuple):
        pad_h = padding[0]
        pad_w = padding[1]
        
        out_height = int((images.shape[1] + (2 * (pad_h))- kh)/stride[0]) + 1
        out_width = int((images.shape[2] + (2 * (pad_w))- kw)/stride[1]) + 1
        padded_input = np.pad(images, ((0,0), (pad_h, pad_h),
                                           (pad_w, pad_w), (0, 0))) # not sure about this

    output = np.zeros((m, out_height, out_width, nc))
    channels = nc
    # print("this is out height", out_height)
    # print("this is out width", out_width)
    for ch in range(nc):
        for i in range(out_height):
            for j in range(out_width):
                # if (i * stride[0] + kh) - (i * stride[0]) != kernel.shape[0]:
                #     pass
                # if (j * stride[1] + kw) - (j * stride[1]) != kernel.shape[1]:
                #     pass
                # else:
                output[:, i, j, ch] += np.sum(padded_input[:, i * stride[0]:(i*stride[0])+kh, j * stride[1]:(j * stride[1]) + kw, :] * kernels[..., ch], axis=(1, 2, 3))
                # why does the 'diagonal' pattern work here? Shouldn't it be with the step of the range function?
                # calculates the sum over the 2nd and third dimensions, ignoring the first


    return np.array(output)
