#!/usr/bin/env python3
"""simple BiRNN made in Numpy"""
import numpy as np


class BidirectionalCell:
    """BiRNN cell in numpy"""

    def __init__(self, i, h, o):
        """init function"""
        self.i = i
        self.h = h
        self.o = o

        # initizialing weights
        self.Whf = np.random.normal(0, 1, size=(i + h, h))
        self.bhf = np.expand_dims(np.zeros(h,), axis=0)

        self.Whb = np.random.normal(0, 1, size=(i + h, h))
        self.bhb = np.expand_dims(np.zeros(h,), axis=-0)

        self.Wy = np.random.normal(0, 1, size=(h + i + o, o))
        self.by = np.expand_dims(np.zeros(o,), axis=0)

    def forward(self, h_prev, x_t):
        """a single forward pass of the BiRNN cell
        does not include the backward pass"""
        h_concat = np.concatenate((h_prev, x_t), axis=-1)

        h_concat_f = np.matmul(h_concat, self.Whf) + self.bhf

        h_next = np.tanh(h_concat_f)
        return h_next

    def backward(self, h_next, x_t):
        """a single backward pass of the BiRNN cell
        does not include the forward pass"""
        h_concat = np.concatenate((h_next, x_t), axis=-1)

        h_concat_f = np.matmul(h_concat, self.Whb) + self.bhb

        h_prev = np.tanh(h_concat_f)
        return h_prev
