#!/usr/bin/env python3
"""performs gensin Word2Vec model"""
import tensorflow as tf


def gensim_to_keras(model):
    """converts gensim word2vec model to keras model"""
    # embedding_matrix = model.wv.vectors
    # vocab_size, embedding_dim = embedding_matrix.shape
    #
    # embedding_layer = tf.keras.layers.Embedding(
    #     input_dim=vocab_size,
    #     output_dim=embedding_dim,
    #     weights=[embedding_matrix],
    #     trainable=True,
    # )
    #
    # return embedding_layer

    vocab_size = len(model.wv)
    vector_size = model.wv.vector_size

    embedding_matrix = tf.Variable(tf.zeros((vocab_size + 1, vector_size)), trainable=False)

    for i, word in enumerate(model.wv.index_to_key):
        embedding_matrix = embedding_matrix[i + 1].assign(model.wv[word])

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix.numpy()),
        trainable=True
    )
    return embedding_layer
