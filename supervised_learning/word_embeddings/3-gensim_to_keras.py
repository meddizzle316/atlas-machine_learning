#!/usr/bin/env python3
"""performs gensin Word2Vec model"""
import tensorflow as tf


def gensim_to_keras(model):
    """converts gensim word2vec model to keras model"""
    embedding_matrix = model.wv.vectors
    vocab_size, embedding_dim = embedding_matrix.shape


    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True,
    )

    return embedding_layer
