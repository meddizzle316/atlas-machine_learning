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
    vocab_size = len(model.wv.index_to_key)  # Number of words in the vocabulary
    embedding_dim = model.vector_size  # Dimensionality of embeddings

    # Extract weights from the gensim Word2Vec model
    weights = model.wv.vectors  # A numpy array of shape (vocab_size, embedding_dim)

    # Create a trainable Keras Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,  # Vocabulary size
        output_dim=embedding_dim,  # Size of each embedding vector
        weights=[weights],  # Pre-trained weights
        trainable=True  # Allow fine-tuning of embeddings
    )

    return embedding_layer