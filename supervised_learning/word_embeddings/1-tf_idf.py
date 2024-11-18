#!/usr/bin/env python3
"""performs basic tf_idf"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """does tf_idf for each sentence with sklearn"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = tfidf_matrix.toarray()
    return tfidf_matrix, feature_names