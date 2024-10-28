#!/usr/bin/env python3
"""makes simple rnn unit"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        self.i = i
        self.o = o
        self.h = h

        # public instance attributes
        self.Wh = np.random.normal(0, 1, (int(i + h), h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bh = np.expand_dims(np.zeros((h,)), 0)
        self.by = np.expand_dims(np.zeros((o,)), axis=0)


    def forward(self, h_prev, x_t):
        """performs the forward pass of the RNN cell
        returns h_next, y"""

        concat = np.concatenate([h_prev, x_t], axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y