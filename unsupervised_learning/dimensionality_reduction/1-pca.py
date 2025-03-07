#!/usr/bin/env python3
"""performs pca on given nd array"""
import numpy as np


def pca(X, ndim):
    """performs pcs on given nd array
    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA trans should
    maintain

    Returns: weights matrix, W, has var fraction of X‘s original variance
    W is a numpy.ndarray of shape (d, nd) where
    nd is the new dimensionality of the transformed X"""

    n, d = X.shape
    # standardize data
    # either that or X - X.mean(), not sure which to use
    X = (X - X.mean(axis=0))

    # get covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # get new number of components

    explained_variance = eigenvalues / np.sum(eigenvalues)

    reduced_data = np.matmul(X, eigenvectors[:, :ndim])

    return np.real(reduced_data)
