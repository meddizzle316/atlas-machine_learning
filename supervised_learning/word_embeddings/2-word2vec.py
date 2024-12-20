#!/usr/bin/env python3
"""performs gensin Word2Vec model"""
import gensim


def word2vec_model(sentences, vector_size=100,
                   min_count=5, window=5, negative=5,
                   cbow=True, epochs=5, seed=0,
                   workers=1):
    """makes a word2vec model"""
    sg = 1 if not cbow else 0

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        workers=workers,
        window=window,
        seed=seed,
        negative=negative,
        epochs=epochs,
        sg=sg
    )

    model.train(sentences, total_examples=1,
                epochs=model.epochs, compute_loss=True)
    return model
