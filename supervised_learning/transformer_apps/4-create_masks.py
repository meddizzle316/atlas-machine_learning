#!/usr/bin/env python3
"""function to create masks"""
import tensorflow as tf


def create_masks(inputs, targets):
    """creates alls masks for training/evaluation"""
    batch_size, seq_len_in = inputs.shape
    _, seq_len_out = targets.shape
    en_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    en_mask = en_mask[:, tf.newaxis, tf.newaxis, :]
    # print("en mask shape", en_mask.shape)
    de_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)

    de_mask = de_mask[:, tf.newaxis, tf.newaxis, :]
    # print("de mask shape", de_mask.shape)

    look_ahead = 1 - \
        tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)
    # print("look ahead shape", look_ahead.shape)

    padding_mask = tf.cast(tf.math.equal(targets, 0), tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead, padding_mask)
    return en_mask, combined_mask, de_mask
