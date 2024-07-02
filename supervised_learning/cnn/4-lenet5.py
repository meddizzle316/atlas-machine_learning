#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""builds modified version of LeNet-5
trying to add more documentation
using old tf version like a dingus
"""


def lenet5(x, y):
    """builds modified version of LeNet-5
    uses tf1 but honestly looks a lot
    like tf2"""
    m, h, w, c = x.shape

    he = tf.keras.initializers.VarianceScaling(scale=2.0)

    first_conv_layer = tf.layers.Conv2D(filters=6,
                                        kernel_size=5,
                                        input_shape=(m, h, w, c),
                                        padding='same',
                                        activation='relu',
                                        kernel_initializer=he)(x)
    first_pool = tf.layers.MaxPooling2D(pool_size=2,
                                        strides=2)(first_conv_layer)
    second_conv_layer = tf.layers.Conv2D(filters=16,
                                         kernel_size=5,
                                         padding='valid',
                                         activation='relu',
                                         kernel_initializer=he)(first_pool)
    second_pool_layer = tf.layers.MaxPooling2D(pool_size=2,
                                               strides=2)(second_conv_layer)
    flatten_layer = tf.layers.Flatten()(second_pool_layer)
    first_dense_layer = tf.layers.Dense(120,
                                        kernel_initializer=he,
                                        activation='relu')(flatten_layer)
    second_dense_l = tf.layers.Dense(84, kernel_initializer=he,
                                     activation='relu')(first_dense_layer)
    third_dense_layer = tf.layers.Dense(10,
                                        kernel_initializer=he)(second_dense_l)

    loss = tf.losses.softmax_cross_entropy(y, third_dense_layer)

    opt = tf.train.AdamOptimizer().minimize(loss)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(third_dense_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    softmax = tf.nn.softmax(third_dense_layer)

    return softmax, opt, loss, accuracy
