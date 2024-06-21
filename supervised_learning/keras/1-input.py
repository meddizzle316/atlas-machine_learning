#!/usr/bin/env python3
"""building model with keras api"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """building sequential model without using 
    sequential class"""

    l2 = K.regularizers.l2(l2=lambtha)
    input = K.layers.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=l2)(input)
    if len(layers) == 1:
        return K.models.Model(input, x)
    x = K.layers.Dropout(1 - keep_prob)(x)
    for i in range(1, len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=l2)(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    # I think because I was 'redefining' i (changed to inputs)
    # in line 12 to an integer in line 14, every time I referenced
    # i after refered to the integer created by the for loop
    # instead of the inputs, didn't rea
    return K.models.Model(input, x)
