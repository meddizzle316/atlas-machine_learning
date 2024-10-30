#!/usr/bin/env python3
"""basic LSTM model in numpy"""
import numpy as np


class LSTMCell:
    """the LSTM cell class"""

    def __init__(self, i, h, o):
        """init function"""
        self.i = i
        self.h = h
        self.o = o

        # new public instance variables
        self.Wf = np.random.normal(0, 1, size=(i + h, h))
        self.bf = np.expand_dims(np.zeros(h,), axis=0)

        self.Wu = np.random.normal(0, 1, size=(i + h, h))
        self.bu = np.expand_dims(np.zeros(h, ), axis=0)

        self.Wc = np.random.normal(0, 1, size=(i + h, h))
        self.bc = np.expand_dims(np.zeros(h, ), axis=0)

        self.Wo = np.random.normal(0, 1, size=(i + h, h))
        self.bo = np.expand_dims(np.zeros(h, ), axis=0)

        self.Wy = np.random.normal(0, 1, size=(h, o))
        self.by = np.expand_dims(np.zeros(o, ), axis=0)

    def forward(self, h_prev, c_prev, x_t):
        """one forward pass, returns
        h_next
        c_next
        y"""

        h_x_stack = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(h_x_stack, self.Wf) + self.bf)

        i = self.sigmoid(np.matmul(h_x_stack, self.Wu) + self.bu)

        c = np.tanh(np.matmul(h_x_stack, self.Wc) + self.bc)

        c_prev = f * c_prev + i * c

        o = self.sigmoid(np.dot(h_x_stack, self.Wo) + self.bo)

        h_next = o * np.tanh(c_prev)

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_prev, y

    def sigmoid(self, z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """softmax function"""
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
