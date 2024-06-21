#!/usr/bin/env python3
"""making a sequential model"""
import tensorflow.keras as K


# def build_model(nx, layers, activations, lambtha, keep_prob):
#     """builds sequential tf model"""
#     print(f"this is nx {nx}") is 784
#     print(f"this is the shape of nx {nx.shape}") just int obj, not numpy

#     nx_list = [nx, 1]
#     l2 = K.regularizers.l2(l2=lambtha)
#     first = K.layers.Dense(layers[0], activations[0], kernel_regularizer=l2, input_shape=[2])(nx_list)
#     x = K.layers.BatchNormalization(1 - keep_prob)(first)
#     for i in range(1, len(layers)):
#         x = K.layers.Dense(layers[i], activations[i], kernel_regularizer=l2)(x)
#         x = K.layers.BatchNormalization(1 - keep_prob)(x)
#     return K.models.Sequential(first, x)


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds sequential keras model"""

    model = K.models.Sequential()
    l2 = K.regularizers.l2(l2=lambtha)
    model.add(K.layers.Dense(layers[0], 
                                activation=activations[0],
                                kernel_regularizer=l2,
                                input_shape=(nx,)))
    model.add(K.layers.Dropout(1 - keep_prob))
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], 
                                activation=activations[i],
                                kernel_regularizer=l2))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
    