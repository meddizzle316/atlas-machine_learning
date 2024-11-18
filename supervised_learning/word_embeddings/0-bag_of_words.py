#!/usr/bin/env python3
"""performs basic bag of words"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def bag_of_words(sentences, vocab=None):
    """basic function that uses sklearn to do bag of words"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    X = X.toarray()
    return X, feature_names
