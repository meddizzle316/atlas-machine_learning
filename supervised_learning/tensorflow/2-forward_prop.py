#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """does forward prop"""

    current_layer = x
    for i in range(len(layer_sizes)):
        num_nodes = layer_sizes[i]
        activation_func = activations[i]

        current_layer = create_layer(current_layer, num_nodes, activation_func)

    return current_layer
