#!/usr/bin/env python
"""intra variance of dataset"""
import numpy as np

def initialize(X, k):
    """initialize cluster centriods for k-means
        X is a numpy.ndarray of shape (n, d) containing the dataset
        that will be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters

        The cluster centroids should be initialized with a multivariate
        uniform distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
        values of X along each dimension in d
        The maximum values for the distribution should be the maximum
        values of X along each dimension in d
        You should use numpy.random.uniform exactly once
        You are not allowed to use any loops

        Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(k, int):
        return None
    if k <= 0:
        return None
    if not isinstance(X, np.ndarray) or X.ndim < 2:
        return None
    try:
        m, d = X.shape
    except ValueError:
        return None
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    centroids = np.random.uniform(min_val, max_val, size=(k, d))
    return centroids


def kmeans(X, k, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the dataset
           n is the number of data points
           d is the number of dimensions of each data point
       k is a positive integer containing the number of clusters
       iterations is a positive integer containing the maximum
       number of iterations that should be performed
       If no change in the cluster centroids occurs between
       iterations, your function should return
       Initialize the cluster centroids using a multivariate
       uniform distribution (based on0-initialize.py)
       If a cluster contains no data points during the update
       step, reinitialize its centroid
       You should use numpy.random.uniform exactly twice
       You may use at most 2 loops

       Returns: C, clss, or None, None on failure
           C is a numpy.ndarray of shape (k, d) containing the
           centroid means for each cluster
           clss is a numpy.ndarray of shape (n,) containing the
           index of the cluster in C that each data point belongs to"""

    if not isinstance(k, int):
        return None, None
    if k <= 0:
        return None, None
    if not isinstance(X, np.ndarray) or X.ndim < 2:
        return None, None
    if not isinstance(iterations, int):
        return None, None
    if iterations <= 0:
        return None, None
    try:
        m, d = X.shape
    except ValueError:
        return None, None

    centroids = initialize(X, k)
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    labels = np.random.randint(low=0, high=k, size=m)

    for i in range(iterations):
        distances = np.linalg.norm(
            np.expand_dims(
                X,
                2) -
            np.expand_dims(
                centroids.T,
                0),
            axis=1)
        new_labels = np.argmin(distances, axis=1)

        if (labels == new_labels).all():
            labels = new_labels
            break

        else:
            difference = np.mean(labels != new_labels)
            labels = new_labels
            for c in range(k):
                centroids[c] = np.mean(X[labels == c], axis=0)
                if np.any(np.isnan(centroids[c])):
                    new_clusters = np.random.uniform(
                        min_val, max_val, size=centroids[c].shape)
                    centroids[c] = new_clusters

    return centroids, labels
def variance(X, C):
    """X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
         You are not allowed to use any loops
        Returns: var, or None on failure
        var is the total variance"""

    # Calculate squared distances
    # Assign each data point to its nearest centroid
    distances = np.min(np.linalg.norm(X[:, None] - C, axis=2), axis=1)

    # Calculate the squared distances
    squared_distances = distances ** 2

    # Calculate the total intra-cluster variance
    variance = np.sum(squared_distances)

    return variance

