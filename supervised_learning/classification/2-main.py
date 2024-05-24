#!/usr/bin/env python3

import numpy as np

Neuron = __import__('2-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
# Shape of given Neuron is (1,784) 
# Shape of X is (784,12665) 
print(f"The shape is {X.shape[0]}")
print(f"X is {X}")
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)
