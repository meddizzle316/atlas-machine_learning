#!/usr/bin/env python3
"""simple class for GRU network in numpy"""
import numpy as np


class GRUCell:
    """simple class for GRU network in numpy"""

    def __init__(self, i, h, o):
        """init function"""
        self.i = i  # 10
        self.h = h  # 15
        self.o = o  # 5

        # new public instance attributes
        self.Wz = np.random.normal(0, 1, size=(i + h, h))
        self.bz = np.expand_dims(np.zeros(h,), axis=0)

        self.Wr = np.random.normal(0, 1, size=(i + h, h))
        self.br = np.expand_dims(np.zeros(h,), axis=0)

        self.Wh = np.random.normal(0, 1, size=(i + h, h))
        self.bh = np.expand_dims(np.zeros(h,), axis=0)

        self.Wy = np.random.normal(0, 1, size=(h, o))
        self.by = np.expand_dims(np.zeros(o,), axis=0)

    def forward(self, h_prev, x_t):
        """forward function"""

        concat = np.concatenate([h_prev, x_t], axis=1)

        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        r_tilde = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(r_tilde, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """softmax function"""
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
