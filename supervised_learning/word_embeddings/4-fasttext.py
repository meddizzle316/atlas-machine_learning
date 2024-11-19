#!/usr/bin/env python3
"""builds and trains a basic gensim fasttext model"""
import gensim


def fasttext_model(sentences,
                   vector_size=100,
                   min_count=5, negative=5,
                   window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """builds and trains a basic gensim fasttext model"""
    sg = 1 if not cbow else 0
    model = gensim.models.FastText(sentences=sentences,
                                   min_count=min_count,
                                   window=window,
                                   vector_size=vector_size,
                                   sg=sg,
                                   epochs=epochs,
                                   seed=seed,
                                   workers=workers,
                                   negative=negative
                                   )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model
